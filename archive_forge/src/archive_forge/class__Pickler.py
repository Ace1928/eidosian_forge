from types import FunctionType
from copyreg import dispatch_table
from copyreg import _extension_registry, _inverted_registry, _extension_cache
from itertools import islice
from functools import partial
import sys
from sys import maxsize
from struct import pack, unpack
import re
import io
import codecs
import _compat_pickle
class _Pickler:

    def __init__(self, file, protocol=None, *, fix_imports=True, buffer_callback=None):
        """This takes a binary file for writing a pickle data stream.

        The optional *protocol* argument tells the pickler to use the
        given protocol; supported protocols are 0, 1, 2, 3, 4 and 5.
        The default protocol is 4. It was introduced in Python 3.4, and
        is incompatible with previous versions.

        Specifying a negative protocol version selects the highest
        protocol version supported.  The higher the protocol used, the
        more recent the version of Python needed to read the pickle
        produced.

        The *file* argument must have a write() method that accepts a
        single bytes argument. It can thus be a file object opened for
        binary writing, an io.BytesIO instance, or any other custom
        object that meets this interface.

        If *fix_imports* is True and *protocol* is less than 3, pickle
        will try to map the new Python 3 names to the old module names
        used in Python 2, so that the pickle data stream is readable
        with Python 2.

        If *buffer_callback* is None (the default), buffer views are
        serialized into *file* as part of the pickle stream.

        If *buffer_callback* is not None, then it can be called any number
        of times with a buffer view.  If the callback returns a false value
        (such as None), the given buffer is out-of-band; otherwise the
        buffer is serialized in-band, i.e. inside the pickle stream.

        It is an error if *buffer_callback* is not None and *protocol*
        is None or smaller than 5.
        """
        if protocol is None:
            protocol = DEFAULT_PROTOCOL
        if protocol < 0:
            protocol = HIGHEST_PROTOCOL
        elif not 0 <= protocol <= HIGHEST_PROTOCOL:
            raise ValueError('pickle protocol must be <= %d' % HIGHEST_PROTOCOL)
        if buffer_callback is not None and protocol < 5:
            raise ValueError('buffer_callback needs protocol >= 5')
        self._buffer_callback = buffer_callback
        try:
            self._file_write = file.write
        except AttributeError:
            raise TypeError("file must have a 'write' attribute")
        self.framer = _Framer(self._file_write)
        self.write = self.framer.write
        self._write_large_bytes = self.framer.write_large_bytes
        self.memo = {}
        self.proto = int(protocol)
        self.bin = protocol >= 1
        self.fast = 0
        self.fix_imports = fix_imports and protocol < 3

    def clear_memo(self):
        """Clears the pickler's "memo".

        The memo is the data structure that remembers which objects the
        pickler has already seen, so that shared or recursive objects
        are pickled by reference and not by value.  This method is
        useful when re-using picklers.
        """
        self.memo.clear()

    def dump(self, obj):
        """Write a pickled representation of obj to the open file."""
        if not hasattr(self, '_file_write'):
            raise PicklingError('Pickler.__init__() was not called by %s.__init__()' % (self.__class__.__name__,))
        if self.proto >= 2:
            self.write(PROTO + pack('<B', self.proto))
        if self.proto >= 4:
            self.framer.start_framing()
        self.save(obj)
        self.write(STOP)
        self.framer.end_framing()

    def memoize(self, obj):
        """Store an object in the memo."""
        if self.fast:
            return
        assert id(obj) not in self.memo
        idx = len(self.memo)
        self.write(self.put(idx))
        self.memo[id(obj)] = (idx, obj)

    def put(self, idx):
        if self.proto >= 4:
            return MEMOIZE
        elif self.bin:
            if idx < 256:
                return BINPUT + pack('<B', idx)
            else:
                return LONG_BINPUT + pack('<I', idx)
        else:
            return PUT + repr(idx).encode('ascii') + b'\n'

    def get(self, i):
        if self.bin:
            if i < 256:
                return BINGET + pack('<B', i)
            else:
                return LONG_BINGET + pack('<I', i)
        return GET + repr(i).encode('ascii') + b'\n'

    def save(self, obj, save_persistent_id=True):
        self.framer.commit_frame()
        pid = self.persistent_id(obj)
        if pid is not None and save_persistent_id:
            self.save_pers(pid)
            return
        x = self.memo.get(id(obj))
        if x is not None:
            self.write(self.get(x[0]))
            return
        rv = NotImplemented
        reduce = getattr(self, 'reducer_override', None)
        if reduce is not None:
            rv = reduce(obj)
        if rv is NotImplemented:
            t = type(obj)
            f = self.dispatch.get(t)
            if f is not None:
                f(self, obj)
                return
            reduce = getattr(self, 'dispatch_table', dispatch_table).get(t)
            if reduce is not None:
                rv = reduce(obj)
            else:
                if issubclass(t, type):
                    self.save_global(obj)
                    return
                reduce = getattr(obj, '__reduce_ex__', None)
                if reduce is not None:
                    rv = reduce(self.proto)
                else:
                    reduce = getattr(obj, '__reduce__', None)
                    if reduce is not None:
                        rv = reduce()
                    else:
                        raise PicklingError("Can't pickle %r object: %r" % (t.__name__, obj))
        if isinstance(rv, str):
            self.save_global(obj, rv)
            return
        if not isinstance(rv, tuple):
            raise PicklingError('%s must return string or tuple' % reduce)
        l = len(rv)
        if not 2 <= l <= 6:
            raise PicklingError('Tuple returned by %s must have two to six elements' % reduce)
        self.save_reduce(*rv, obj=obj)

    def persistent_id(self, obj):
        return None

    def save_pers(self, pid):
        if self.bin:
            self.save(pid, save_persistent_id=False)
            self.write(BINPERSID)
        else:
            try:
                self.write(PERSID + str(pid).encode('ascii') + b'\n')
            except UnicodeEncodeError:
                raise PicklingError('persistent IDs in protocol 0 must be ASCII strings')

    def save_reduce(self, func, args, state=None, listitems=None, dictitems=None, state_setter=None, *, obj=None):
        if not isinstance(args, tuple):
            raise PicklingError('args from save_reduce() must be a tuple')
        if not callable(func):
            raise PicklingError('func from save_reduce() must be callable')
        save = self.save
        write = self.write
        func_name = getattr(func, '__name__', '')
        if self.proto >= 2 and func_name == '__newobj_ex__':
            cls, args, kwargs = args
            if not hasattr(cls, '__new__'):
                raise PicklingError('args[0] from {} args has no __new__'.format(func_name))
            if obj is not None and cls is not obj.__class__:
                raise PicklingError('args[0] from {} args has the wrong class'.format(func_name))
            if self.proto >= 4:
                save(cls)
                save(args)
                save(kwargs)
                write(NEWOBJ_EX)
            else:
                func = partial(cls.__new__, cls, *args, **kwargs)
                save(func)
                save(())
                write(REDUCE)
        elif self.proto >= 2 and func_name == '__newobj__':
            cls = args[0]
            if not hasattr(cls, '__new__'):
                raise PicklingError('args[0] from __newobj__ args has no __new__')
            if obj is not None and cls is not obj.__class__:
                raise PicklingError('args[0] from __newobj__ args has the wrong class')
            args = args[1:]
            save(cls)
            save(args)
            write(NEWOBJ)
        else:
            save(func)
            save(args)
            write(REDUCE)
        if obj is not None:
            if id(obj) in self.memo:
                write(POP + self.get(self.memo[id(obj)][0]))
            else:
                self.memoize(obj)
        if listitems is not None:
            self._batch_appends(listitems)
        if dictitems is not None:
            self._batch_setitems(dictitems)
        if state is not None:
            if state_setter is None:
                save(state)
                write(BUILD)
            else:
                save(state_setter)
                save(obj)
                save(state)
                write(TUPLE2)
                write(REDUCE)
                write(POP)
    dispatch = {}

    def save_none(self, obj):
        self.write(NONE)
    dispatch[type(None)] = save_none

    def save_bool(self, obj):
        if self.proto >= 2:
            self.write(NEWTRUE if obj else NEWFALSE)
        else:
            self.write(TRUE if obj else FALSE)
    dispatch[bool] = save_bool

    def save_long(self, obj):
        if self.bin:
            if obj >= 0:
                if obj <= 255:
                    self.write(BININT1 + pack('<B', obj))
                    return
                if obj <= 65535:
                    self.write(BININT2 + pack('<H', obj))
                    return
            if -2147483648 <= obj <= 2147483647:
                self.write(BININT + pack('<i', obj))
                return
        if self.proto >= 2:
            encoded = encode_long(obj)
            n = len(encoded)
            if n < 256:
                self.write(LONG1 + pack('<B', n) + encoded)
            else:
                self.write(LONG4 + pack('<i', n) + encoded)
            return
        if -2147483648 <= obj <= 2147483647:
            self.write(INT + repr(obj).encode('ascii') + b'\n')
        else:
            self.write(LONG + repr(obj).encode('ascii') + b'L\n')
    dispatch[int] = save_long

    def save_float(self, obj):
        if self.bin:
            self.write(BINFLOAT + pack('>d', obj))
        else:
            self.write(FLOAT + repr(obj).encode('ascii') + b'\n')
    dispatch[float] = save_float

    def save_bytes(self, obj):
        if self.proto < 3:
            if not obj:
                self.save_reduce(bytes, (), obj=obj)
            else:
                self.save_reduce(codecs.encode, (str(obj, 'latin1'), 'latin1'), obj=obj)
            return
        n = len(obj)
        if n <= 255:
            self.write(SHORT_BINBYTES + pack('<B', n) + obj)
        elif n > 4294967295 and self.proto >= 4:
            self._write_large_bytes(BINBYTES8 + pack('<Q', n), obj)
        elif n >= self.framer._FRAME_SIZE_TARGET:
            self._write_large_bytes(BINBYTES + pack('<I', n), obj)
        else:
            self.write(BINBYTES + pack('<I', n) + obj)
        self.memoize(obj)
    dispatch[bytes] = save_bytes

    def save_bytearray(self, obj):
        if self.proto < 5:
            if not obj:
                self.save_reduce(bytearray, (), obj=obj)
            else:
                self.save_reduce(bytearray, (bytes(obj),), obj=obj)
            return
        n = len(obj)
        if n >= self.framer._FRAME_SIZE_TARGET:
            self._write_large_bytes(BYTEARRAY8 + pack('<Q', n), obj)
        else:
            self.write(BYTEARRAY8 + pack('<Q', n) + obj)
        self.memoize(obj)
    dispatch[bytearray] = save_bytearray
    if _HAVE_PICKLE_BUFFER:

        def save_picklebuffer(self, obj):
            if self.proto < 5:
                raise PicklingError('PickleBuffer can only pickled with protocol >= 5')
            with obj.raw() as m:
                if not m.contiguous:
                    raise PicklingError('PickleBuffer can not be pickled when pointing to a non-contiguous buffer')
                in_band = True
                if self._buffer_callback is not None:
                    in_band = bool(self._buffer_callback(obj))
                if in_band:
                    if m.readonly:
                        self.save_bytes(m.tobytes())
                    else:
                        self.save_bytearray(m.tobytes())
                else:
                    self.write(NEXT_BUFFER)
                    if m.readonly:
                        self.write(READONLY_BUFFER)
        dispatch[PickleBuffer] = save_picklebuffer

    def save_str(self, obj):
        if self.bin:
            encoded = obj.encode('utf-8', 'surrogatepass')
            n = len(encoded)
            if n <= 255 and self.proto >= 4:
                self.write(SHORT_BINUNICODE + pack('<B', n) + encoded)
            elif n > 4294967295 and self.proto >= 4:
                self._write_large_bytes(BINUNICODE8 + pack('<Q', n), encoded)
            elif n >= self.framer._FRAME_SIZE_TARGET:
                self._write_large_bytes(BINUNICODE + pack('<I', n), encoded)
            else:
                self.write(BINUNICODE + pack('<I', n) + encoded)
        else:
            tmp = obj.replace('\\', '\\u005c')
            tmp = tmp.replace('\x00', '\\u0000')
            tmp = tmp.replace('\n', '\\u000a')
            tmp = tmp.replace('\r', '\\u000d')
            tmp = tmp.replace('\x1a', '\\u001a')
            self.write(UNICODE + tmp.encode('raw-unicode-escape') + b'\n')
        self.memoize(obj)
    dispatch[str] = save_str

    def save_tuple(self, obj):
        if not obj:
            if self.bin:
                self.write(EMPTY_TUPLE)
            else:
                self.write(MARK + TUPLE)
            return
        n = len(obj)
        save = self.save
        memo = self.memo
        if n <= 3 and self.proto >= 2:
            for element in obj:
                save(element)
            if id(obj) in memo:
                get = self.get(memo[id(obj)][0])
                self.write(POP * n + get)
            else:
                self.write(_tuplesize2code[n])
                self.memoize(obj)
            return
        write = self.write
        write(MARK)
        for element in obj:
            save(element)
        if id(obj) in memo:
            get = self.get(memo[id(obj)][0])
            if self.bin:
                write(POP_MARK + get)
            else:
                write(POP * (n + 1) + get)
            return
        write(TUPLE)
        self.memoize(obj)
    dispatch[tuple] = save_tuple

    def save_list(self, obj):
        if self.bin:
            self.write(EMPTY_LIST)
        else:
            self.write(MARK + LIST)
        self.memoize(obj)
        self._batch_appends(obj)
    dispatch[list] = save_list
    _BATCHSIZE = 1000

    def _batch_appends(self, items):
        save = self.save
        write = self.write
        if not self.bin:
            for x in items:
                save(x)
                write(APPEND)
            return
        it = iter(items)
        while True:
            tmp = list(islice(it, self._BATCHSIZE))
            n = len(tmp)
            if n > 1:
                write(MARK)
                for x in tmp:
                    save(x)
                write(APPENDS)
            elif n:
                save(tmp[0])
                write(APPEND)
            if n < self._BATCHSIZE:
                return

    def save_dict(self, obj):
        if self.bin:
            self.write(EMPTY_DICT)
        else:
            self.write(MARK + DICT)
        self.memoize(obj)
        self._batch_setitems(obj.items())
    dispatch[dict] = save_dict
    if PyStringMap is not None:
        dispatch[PyStringMap] = save_dict

    def _batch_setitems(self, items):
        save = self.save
        write = self.write
        if not self.bin:
            for k, v in items:
                save(k)
                save(v)
                write(SETITEM)
            return
        it = iter(items)
        while True:
            tmp = list(islice(it, self._BATCHSIZE))
            n = len(tmp)
            if n > 1:
                write(MARK)
                for k, v in tmp:
                    save(k)
                    save(v)
                write(SETITEMS)
            elif n:
                k, v = tmp[0]
                save(k)
                save(v)
                write(SETITEM)
            if n < self._BATCHSIZE:
                return

    def save_set(self, obj):
        save = self.save
        write = self.write
        if self.proto < 4:
            self.save_reduce(set, (list(obj),), obj=obj)
            return
        write(EMPTY_SET)
        self.memoize(obj)
        it = iter(obj)
        while True:
            batch = list(islice(it, self._BATCHSIZE))
            n = len(batch)
            if n > 0:
                write(MARK)
                for item in batch:
                    save(item)
                write(ADDITEMS)
            if n < self._BATCHSIZE:
                return
    dispatch[set] = save_set

    def save_frozenset(self, obj):
        save = self.save
        write = self.write
        if self.proto < 4:
            self.save_reduce(frozenset, (list(obj),), obj=obj)
            return
        write(MARK)
        for item in obj:
            save(item)
        if id(obj) in self.memo:
            write(POP_MARK + self.get(self.memo[id(obj)][0]))
            return
        write(FROZENSET)
        self.memoize(obj)
    dispatch[frozenset] = save_frozenset

    def save_global(self, obj, name=None):
        write = self.write
        memo = self.memo
        if name is None:
            name = getattr(obj, '__qualname__', None)
        if name is None:
            name = obj.__name__
        module_name = whichmodule(obj, name)
        try:
            __import__(module_name, level=0)
            module = sys.modules[module_name]
            obj2, parent = _getattribute(module, name)
        except (ImportError, KeyError, AttributeError):
            raise PicklingError("Can't pickle %r: it's not found as %s.%s" % (obj, module_name, name)) from None
        else:
            if obj2 is not obj:
                raise PicklingError("Can't pickle %r: it's not the same object as %s.%s" % (obj, module_name, name))
        if self.proto >= 2:
            code = _extension_registry.get((module_name, name))
            if code:
                assert code > 0
                if code <= 255:
                    write(EXT1 + pack('<B', code))
                elif code <= 65535:
                    write(EXT2 + pack('<H', code))
                else:
                    write(EXT4 + pack('<i', code))
                return
        lastname = name.rpartition('.')[2]
        if parent is module:
            name = lastname
        if self.proto >= 4:
            self.save(module_name)
            self.save(name)
            write(STACK_GLOBAL)
        elif parent is not module:
            self.save_reduce(getattr, (parent, lastname))
        elif self.proto >= 3:
            write(GLOBAL + bytes(module_name, 'utf-8') + b'\n' + bytes(name, 'utf-8') + b'\n')
        else:
            if self.fix_imports:
                r_name_mapping = _compat_pickle.REVERSE_NAME_MAPPING
                r_import_mapping = _compat_pickle.REVERSE_IMPORT_MAPPING
                if (module_name, name) in r_name_mapping:
                    module_name, name = r_name_mapping[module_name, name]
                elif module_name in r_import_mapping:
                    module_name = r_import_mapping[module_name]
            try:
                write(GLOBAL + bytes(module_name, 'ascii') + b'\n' + bytes(name, 'ascii') + b'\n')
            except UnicodeEncodeError:
                raise PicklingError("can't pickle global identifier '%s.%s' using pickle protocol %i" % (module, name, self.proto)) from None
        self.memoize(obj)

    def save_type(self, obj):
        if obj is type(None):
            return self.save_reduce(type, (None,), obj=obj)
        elif obj is type(NotImplemented):
            return self.save_reduce(type, (NotImplemented,), obj=obj)
        elif obj is type(...):
            return self.save_reduce(type, (...,), obj=obj)
        return self.save_global(obj)
    dispatch[FunctionType] = save_global
    dispatch[type] = save_type