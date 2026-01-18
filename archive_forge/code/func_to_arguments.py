from itertools import chain
from graphql import Undefined
from .dynamic import Dynamic
from .mountedtype import MountedType
from .structures import NonNull
from .utils import get_type
def to_arguments(args, extra_args=None):
    from .unmountedtype import UnmountedType
    from .field import Field
    from .inputfield import InputField
    if extra_args:
        extra_args = sorted(extra_args.items(), key=lambda f: f[1])
    else:
        extra_args = []
    iter_arguments = chain(args.items(), extra_args)
    arguments = {}
    for default_name, arg in iter_arguments:
        if isinstance(arg, Dynamic):
            arg = arg.get_type()
            if arg is None:
                continue
        if isinstance(arg, UnmountedType):
            arg = Argument.mounted(arg)
        if isinstance(arg, (InputField, Field)):
            raise ValueError(f'Expected {default_name} to be Argument, but received {type(arg).__name__}. Try using Argument({arg.type}).')
        if not isinstance(arg, Argument):
            raise ValueError(f'Unknown argument "{default_name}".')
        arg_name = default_name or arg.name
        assert arg_name not in arguments, f'More than one Argument have same name "{arg_name}".'
        arguments[arg_name] = arg
    return arguments