import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
class __RL_SAFE_ENV__:
    __time_time__ = time.time
    __weakref_ref__ = weakref.ref
    __slicetype__ = type(slice(0))

    def __init__(self, timeout=None, allowed_magic_methods=None):
        self.timeout = timeout if timeout is not None else self.__rl_tmax__
        self.allowed_magic_methods = (__allowed_magic_methods__ if allowed_magic_methods == True else allowed_magic_methods) if allowed_magic_methods else []
        import builtins
        self.__rl_gen_range__ = builtins.range
        self.__rl_real_iter__ = builtins.iter

        class __rl_dict__(dict):

            def __new__(cls, *args, **kwds):
                if len(args) == 1 and (not isinstance(args[0], dict)):
                    try:
                        it = self.__real_iter__(args[0])
                    except TypeError:
                        pass
                    else:
                        args = (self.__rl_getiter__(it),)
                return dict.__new__(cls, *args, **kwds)

        class __rl_missing_func__:

            def __init__(self, name):
                self.__name__ = name

            def __call__(self, *args, **kwds):
                raise BadCode('missing global %s' % self.__name__)
        self.real_bi = builtins
        self.bi_replace = (('open', __rl_missing_func__('open')), ('iter', self.__rl_getiter__))
        __rl_safe_builtins__.update({_: getattr(builtins, _) for _ in 'None False True abs bool callable chr complex divmod float hash hex id int\n\t\tisinstance issubclass len oct ord range repr round slice str tuple setattr\n\t\tclassmethod staticmethod property divmod next object getattr dict iter pow list\n\t\ttype max min sum enumerate zip hasattr filter map any all sorted reversed range\n\t\tset frozenset\n\n\t\tArithmeticError AssertionError AttributeError BaseException BufferError BytesWarning\n\t\tDeprecationWarning EOFError EnvironmentError Exception FloatingPointError FutureWarning\n\t\tGeneratorExit IOError ImportError ImportWarning IndentationError IndexError KeyError\n\t\tKeyboardInterrupt LookupError MemoryError NameError NotImplementedError OSError\n\t\tOverflowError PendingDeprecationWarning ReferenceError RuntimeError RuntimeWarning\n\t\tStopIteration SyntaxError SyntaxWarning SystemError SystemExit TabError TypeError\n\t\tUnboundLocalError UnicodeDecodeError UnicodeEncodeError UnicodeError UnicodeTranslateError\n\t\tUnicodeWarning UserWarning ValueError Warning ZeroDivisionError\n\t\t__build_class__'.split()})
        self.__rl_builtins__ = __rl_builtins__ = {_: __rl_missing_func__(_) for _ in dir(builtins) if callable(getattr(builtins, _))}
        __rl_builtins__.update(__rl_safe_builtins__)
        __rl_builtins__['__rl_add__'] = self.__rl_add__
        __rl_builtins__['__rl_mult__'] = self.__rl_mult__
        __rl_builtins__['__rl_pow__'] = self.__rl_pow__
        __rl_builtins__['__rl_sd__'] = self.__rl_sd__
        __rl_builtins__['__rl_augAssign__'] = self.__rl_augAssign__
        __rl_builtins__['__rl_getitem__'] = self.__rl_getitem__
        __rl_builtins__['__rl_getattr__'] = self.__rl_getattr__
        __rl_builtins__['__rl_getiter__'] = self.__rl_getiter__
        __rl_builtins__['__rl_max_len__'] = self.__rl_max_len__
        __rl_builtins__['__rl_max_pow_digits__'] = self.__rl_max_pow_digits__
        __rl_builtins__['__rl_iter_unpack_sequence__'] = self.__rl_iter_unpack_sequence__
        __rl_builtins__['__rl_unpack_sequence__'] = self.__rl_unpack_sequence__
        __rl_builtins__['__rl_apply__'] = lambda func, *args, **kwds: self.__rl_apply__(func, args, kwds)
        __rl_builtins__['__rl_SafeIter__'] = __rl_SafeIter__
        __rl_builtins__['getattr'] = self.__rl_getattr__
        __rl_builtins__['dict'] = __rl_dict__
        __rl_builtins__['iter'] = self.__rl_getiter__
        __rl_builtins__['pow'] = self.__rl_pow__
        __rl_builtins__['list'] = self.__rl_list__
        __rl_builtins__['type'] = self.__rl_type__
        __rl_builtins__['max'] = self.__rl_max__
        __rl_builtins__['min'] = self.__rl_min__
        __rl_builtins__['sum'] = self.__rl_sum__
        __rl_builtins__['enumerate'] = self.__rl_enumerate__
        __rl_builtins__['zip'] = self.__rl_zip__
        __rl_builtins__['hasattr'] = self.__rl_hasattr__
        __rl_builtins__['filter'] = self.__rl_filter__
        __rl_builtins__['map'] = self.__rl_map__
        __rl_builtins__['any'] = self.__rl_any__
        __rl_builtins__['all'] = self.__rl_all__
        __rl_builtins__['sorted'] = self.__rl_sorted__
        __rl_builtins__['reversed'] = self.__rl_reversed__
        __rl_builtins__['range'] = self.__rl_range__
        __rl_builtins__['set'] = self.__rl_set__
        __rl_builtins__['frozenset'] = self.__rl_frozenset__

    def __rl_type__(self, *args):
        if len(args) == 1:
            return type(*args)
        raise BadCode('type call error')

    def __rl_check__(self):
        if self.__time_time__() >= self.__rl_limit__:
            raise BadCode('Resources exceeded')

    def __rl_sd__(self, obj):
        return obj

    def __rl_getiter__(self, it):
        return __rl_SafeIter__(it, owner=self.__weakref_ref__(self))

    def __rl_max__(self, arg, *args, **kwds):
        if args:
            arg = [arg]
            arg.extend(args)
        return max(self.__rl_args_iter__(arg), **kwds)

    def __rl_min__(self, arg, *args, **kwds):
        if args:
            arg = [arg]
            arg.extend(args)
        return min(self.__rl_args_iter__(arg), **kwds)

    def __rl_sum__(self, sequence, start=0):
        return sum(self.__rl_args_iter__(sequence), start)

    def __rl_enumerate__(self, seq):
        return enumerate(self.__rl_args_iter__(seq))

    def __rl_zip__(self, *args):
        return zip(*[self.__rl_args_iter__(self.__rl_getitem__(args, i)) for i in range(len(args))])

    def __rl_hasattr__(self, obj, name):
        try:
            self.__rl_getattr__(obj, name)
        except (AttributeError, BadCode, TypeError):
            return False
        return True

    def __rl_filter__(self, f, seq):
        return filter(f, self.__rl_args_iter__(seq))

    def __rl_map__(self, f, seq):
        return map(f, self.__rl_args_iter__(seq))

    def __rl_any__(self, seq):
        return any(self.__rl_args_iter__(seq))

    def __rl_all__(self, seq):
        return all(self.__rl_args_iter__(seq))

    def __rl_sorted__(self, seq, **kwds):
        return sorted(self.__rl_args_iter__(seq), **kwds)

    def __rl_reversed__(self, seq):
        return self.__rl_args_iter__(reversed(seq))

    def __rl_range__(self, start, *args):
        return self.__rl_getiter__(range(start, *args))

    def __rl_set__(self, it):
        return set(self.__rl_args_iter__(it))

    def __rl_frozenset__(self, it):
        return frozenset(self.__rl_args_iter__(it))

    def __rl_iter_unpack_sequence__(self, it, spec, _getiter_):
        """Protect sequence unpacking of targets in a 'for loop'.

		The target of a for loop could be a sequence.
		For example "for a, b in it"
		=> Each object from the iterator needs guarded sequence unpacking.
		"""
        for ob in _getiter_(it):
            yield self.__rl_unpack_sequence__(ob, spec, _getiter_)

    def __rl_unpack_sequence__(self, it, spec, _getiter_):
        """Protect nested sequence unpacking.

		Protect the unpacking of 'it' by wrapping it with '_getiter_'.
		Furthermore for each child element, defined by spec,
		__rl_unpack_sequence__ is called again.

		Have a look at transformer.py 'gen_unpack_spec' for a more detailed
		explanation.
		"""
        ret = list(self.__rl__getiter__(it))
        if len(ret) < spec['min_len']:
            return ret
        for idx, child_spec in spec['childs']:
            ret[idx] = self.__rl_unpack_sequence__(ret[idx], child_spec, _getiter_)
        return ret

    def __rl_is_allowed_name__(self, name):
        """Check names if they are allowed.
		If ``allow_magic_methods is True`` names in `__allowed_magic_methods__`
		are additionally allowed although their names start with `_`.
		"""
        if isinstance(name, strTypes):
            if name in __rl_unsafe__ or (name.startswith('__') and name != '__' and (name not in self.allowed_magic_methods)):
                raise BadCode('unsafe access of %s' % name)

    def __rl_getattr__(self, obj, a, *args):
        if isinstance(obj, strTypes) and a == 'format':
            raise BadCode('%s.format is not implemented' % type(obj))
        self.__rl_is_allowed_name__(a)
        return getattr(obj, a, *args)

    def __rl_getitem__(self, obj, a):
        if type(a) is self.__slicetype__:
            if a.step is not None:
                v = obj[a]
            else:
                start = a.start
                stop = a.stop
                if start is None:
                    start = 0
                if stop is None:
                    v = obj[start:]
                else:
                    v = obj[start:stop]
            return v
        elif isinstance(a, strTypes):
            self.__rl_is_allowed_name__(a)
            return obj[a]
        return obj[a]
    __rl_tmax__ = 5
    __rl_max_len__ = 100000
    __rl_max_pow_digits__ = 100

    def __rl_add__(self, a, b):
        if hasattr(a, '__len__') and hasattr(b, '__len__') and (len(a) + len(b) > self.__rl_max_len__):
            raise BadCode('excessive length')
        return a + b

    def __rl_mult__(self, a, b):
        if hasattr(a, '__len__') and b * len(a) > self.__rl_max_len__ or (hasattr(b, '__len__') and a * len(b) > self.__rl_max_len__):
            raise BadCode('excessive length')
        return a * b

    def __rl_pow__(self, a, b):
        try:
            if b > 0:
                if int(b * math_log10(a) + 1) > self.__rl_max_pow_digits__:
                    raise BadCode
        except:
            raise BadCode('%r**%r invalid or too large' % (a, b))
        return a ** b

    def __rl_augAssign__(self, op, v, i):
        if op == '+=':
            return self.__rl_add__(v, i)
        if op == '-=':
            return v - i
        if op == '*=':
            return self.__rl_mult__(v, i)
        if op == '/=':
            return v / i
        if op == '%=':
            return v % i
        if op == '**=':
            return self.__rl_pow__(v, i)
        if op == '<<=':
            return v << i
        if op == '>>=':
            return v >> i
        if op == '|=':
            return v | i
        if op == '^=':
            return v ^ i
        if op == '&=':
            return v & i
        if op == '//=':
            return v // i

    def __rl_apply__(self, func, args, kwds):
        obj = getattr(func, '__self__', None)
        if obj:
            if isinstance(obj, dict) and func.__name__ in ('pop', 'setdefault', 'get', 'popitem'):
                self.__rl_is_allowed_name__(args[0])
        return func(*[a for a in self.__rl_getiter__(args)], **{k: v for k, v in kwds.items()})

    def __rl_args_iter__(self, *args):
        if len(args) == 1:
            i = args[0]
            if isinstance(i, __rl_SafeIter__):
                return i
            if not isinstance(i, self.__rl_gen_range__):
                return self.__rl_getiter__(i)
        return self.__rl_getiter__(iter(*args))

    def __rl_list__(self, it):
        return list(self.__rl_getiter__(it))

    def __rl_compile__(self, src, fname='<string>', mode='eval', flags=0, inherit=True, visit=None):
        names_seen = {}
        if not visit:
            bcode = compile(src, fname, mode=mode, flags=flags, dont_inherit=not inherit)
        else:
            astc = ast.parse(src, fname, mode)
            if eval_debug > 0:
                print('pre:\n%s\n' % astFormat(astc))
            astc = visit(astc)
            if eval_debug > 0:
                print('post:\n%s\n' % astFormat(astc))
            bcode = compile(astc, fname, mode=mode)
        return (bcode, names_seen)

    def __rl_safe_eval__(self, expr, g, l, mode, timeout=None, allowed_magic_methods=None, __frame_depth__=3):
        bcode, ns = self.__rl_compile__(expr, fname='<string>', mode=mode, flags=0, inherit=True, visit=UntrustedAstTransformer(nameIsAllowed=self.__rl_is_allowed_name__).visit)
        if None in (l, g):
            G = sys._getframe(__frame_depth__)
            L = G.f_locals.copy() if l is None else l
            G = G.f_globals.copy() if g is None else g
        else:
            G = g
            L = l
        obi = (G['__builtins__'],) if '__builtins__' in G else False
        G['__builtins__'] = self.__rl_builtins__
        self.__rl_limit__ = self.__time_time__() + (timeout if timeout is not None else self.timeout)
        if allowed_magic_methods is not None:
            self.allowed_magic_methods = (__allowed_magic_methods__ if allowed_magic_methods == True else allowed_magic_methods) if allowed_magic_methods else []
        sbi = [].append
        bi = self.real_bi
        bir = self.bi_replace
        for n, r in bir:
            sbi(getattr(bi, n))
            setattr(bi, n, r)
        try:
            return eval(bcode, G, L)
        finally:
            sbi = sbi.__self__
            for i, (n, r) in enumerate(bir):
                setattr(bi, n, sbi[i])
            if obi:
                G['__builtins__'] = obi[0]