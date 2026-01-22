import os
from argparse import Action
class check_env(Action):
    """
    Check whether the env var ``PET_{dest}`` exists before defaulting to the given ``default`` value.

    Equivalent to
    ``store_true`` argparse built-in action except that the argument can
    be omitted from the commandline if the env var is present and has a
    non-zero value.

    .. note:: it is redundant to pass ``default=True`` for arguments
              that use this action because a flag should be ``True``
              when present and ``False`` otherwise.

    Example:
    ::

     parser.add_argument("--verbose", action=check_env)

     ./program                                  -> args.verbose=False
     ./program --verbose                        -> args.verbose=True
     PET_VERBOSE=1 ./program           -> args.verbose=True
     PET_VERBOSE=0 ./program           -> args.verbose=False
     PET_VERBOSE=0 ./program --verbose -> args.verbose=True

    Anti-pattern (don't do this):

    ::

     parser.add_argument("--verbose", action=check_env, default=True)

     ./program                                  -> args.verbose=True
     ./program --verbose                        -> args.verbose=True
     PET_VERBOSE=1 ./program           -> args.verbose=True
     PET_VERBOSE=0 ./program           -> args.verbose=False

    """

    def __init__(self, dest, default=False, **kwargs) -> None:
        env_name = f'PET_{dest.upper()}'
        default = bool(int(os.environ.get(env_name, '1' if default else '0')))
        super().__init__(dest=dest, const=True, default=default, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.const)