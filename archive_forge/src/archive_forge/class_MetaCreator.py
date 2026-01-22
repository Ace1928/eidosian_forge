import array
import copy
import copyreg
import warnings
class MetaCreator(type):

    def __new__(cls, name, base, dct):
        return super(MetaCreator, cls).__new__(cls, name, (base,), dct)

    def __init__(cls, name, base, dct):
        dict_inst = {}
        dict_cls = {}
        for obj_name, obj in dct.items():
            if isinstance(obj, type):
                dict_inst[obj_name] = obj
            else:
                dict_cls[obj_name] = obj

        def init_type(self, *args, **kargs):
            """Replace the __init__ function of the new type, in order to
            add attributes that were defined with **kargs to the instance.
            """
            for obj_name, obj in dict_inst.items():
                setattr(self, obj_name, obj())
            if base.__init__ is not object.__init__:
                base.__init__(self, *args, **kargs)
        cls.__init__ = init_type
        cls.reduce_args = (name, base, dct)
        super(MetaCreator, cls).__init__(name, (base,), dict_cls)

    def __reduce__(cls):
        return (meta_create, cls.reduce_args)