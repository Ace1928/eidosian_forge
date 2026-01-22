from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class RunInTerminalRequestArguments(BaseSchema):
    """
    Arguments for 'runInTerminal' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'kind': {'type': 'string', 'enum': ['integrated', 'external'], 'description': 'What kind of terminal to launch.'}, 'title': {'type': 'string', 'description': 'Optional title of the terminal.'}, 'cwd': {'type': 'string', 'description': 'Working directory for the command. For non-empty, valid paths this typically results in execution of a change directory command.'}, 'args': {'type': 'array', 'items': {'type': 'string'}, 'description': 'List of arguments. The first argument is the command to run.'}, 'env': {'type': 'object', 'description': 'Environment key-value pairs that are added to or removed from the default environment.', 'additionalProperties': {'type': ['string', 'null'], 'description': "Proper values must be strings. A value of 'null' removes the variable from the environment."}}}
    __refs__ = set(['env'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, cwd, args, kind=None, title=None, env=None, update_ids_from_dap=False, **kwargs):
        """
        :param string cwd: Working directory for the command. For non-empty, valid paths this typically results in execution of a change directory command.
        :param array args: List of arguments. The first argument is the command to run.
        :param string kind: What kind of terminal to launch.
        :param string title: Optional title of the terminal.
        :param RunInTerminalRequestArgumentsEnv env: Environment key-value pairs that are added to or removed from the default environment.
        """
        self.cwd = cwd
        self.args = args
        self.kind = kind
        self.title = title
        if env is None:
            self.env = RunInTerminalRequestArgumentsEnv()
        else:
            self.env = RunInTerminalRequestArgumentsEnv(update_ids_from_dap=update_ids_from_dap, **env) if env.__class__ != RunInTerminalRequestArgumentsEnv else env
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        cwd = self.cwd
        args = self.args
        if args and hasattr(args[0], 'to_dict'):
            args = [x.to_dict() for x in args]
        kind = self.kind
        title = self.title
        env = self.env
        dct = {'cwd': cwd, 'args': args}
        if kind is not None:
            dct['kind'] = kind
        if title is not None:
            dct['title'] = title
        if env is not None:
            dct['env'] = env.to_dict(update_ids_to_dap=update_ids_to_dap)
        dct.update(self.kwargs)
        return dct