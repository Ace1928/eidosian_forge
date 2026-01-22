from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
class Select(Construct):
    """
    Selects the first matching subconstruct. It will literally try each of
    the subconstructs, until one matches.

    Notes:
    * requires a seekable stream.

    Parameters:
    * name - the name of the construct
    * subcons - the subcons to try (order-sensitive)
    * include_name - a keyword only argument, indicating whether to include
      the name of the selected subcon in the return value of parsing. default
      is false.

    Example:
    Select("foo",
        UBInt64("large"),
        UBInt32("medium"),
        UBInt16("small"),
        UBInt8("tiny"),
    )
    """
    __slots__ = ['subcons', 'include_name']

    def __init__(self, name, *subcons, **kw):
        include_name = kw.pop('include_name', False)
        if kw:
            raise TypeError("the only keyword argument accepted is 'include_name'", kw)
        Construct.__init__(self, name)
        self.subcons = subcons
        self.include_name = include_name
        self._inherit_flags(*subcons)
        self._set_flag(self.FLAG_DYNAMIC)

    def _parse(self, stream, context):
        for sc in self.subcons:
            pos = stream.tell()
            context2 = context.__copy__()
            try:
                obj = sc._parse(stream, context2)
            except ConstructError:
                stream.seek(pos)
            else:
                context.__update__(context2)
                if self.include_name:
                    return (sc.name, obj)
                else:
                    return obj
        raise SelectError('no subconstruct matched')

    def _build(self, obj, stream, context):
        if self.include_name:
            name, obj = obj
            for sc in self.subcons:
                if sc.name == name:
                    sc._build(obj, stream, context)
                    return
        else:
            for sc in self.subcons:
                stream2 = BytesIO()
                context2 = context.__copy__()
                try:
                    sc._build(obj, stream2, context2)
                except Exception:
                    pass
                else:
                    context.__update__(context2)
                    stream.write(stream2.getvalue())
                    return
        raise SelectError('no subconstruct matched', obj)

    def _sizeof(self, context):
        raise SizeofError("can't calculate size")