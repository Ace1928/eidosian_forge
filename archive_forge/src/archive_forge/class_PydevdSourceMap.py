from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class PydevdSourceMap(BaseSchema):
    """
    Information that allows mapping a local line to a remote source/line.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'line': {'type': 'integer', 'description': 'The local line to which the mapping should map to (e.g.: for an ipython notebook this would be the first line of the cell in the file).'}, 'endLine': {'type': 'integer', 'description': 'The end line.'}, 'runtimeSource': {'description': "The path that the user has remotely -- 'source.path' must be specified (e.g.: for an ipython notebook this could be something as '<ipython-input-1-4561234>')", 'type': 'Source'}, 'runtimeLine': {'type': 'integer', 'description': "The remote line to which the mapping should map to (e.g.: for an ipython notebook this would be always 1 as it'd map the start of the cell)."}}
    __refs__ = set(['runtimeSource'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, line, endLine, runtimeSource, runtimeLine, update_ids_from_dap=False, **kwargs):
        """
        :param integer line: The local line to which the mapping should map to (e.g.: for an ipython notebook this would be the first line of the cell in the file).
        :param integer endLine: The end line.
        :param Source runtimeSource: The path that the user has remotely -- 'source.path' must be specified (e.g.: for an ipython notebook this could be something as '<ipython-input-1-4561234>')
        :param integer runtimeLine: The remote line to which the mapping should map to (e.g.: for an ipython notebook this would be always 1 as it'd map the start of the cell).
        """
        self.line = line
        self.endLine = endLine
        if runtimeSource is None:
            self.runtimeSource = Source()
        else:
            self.runtimeSource = Source(update_ids_from_dap=update_ids_from_dap, **runtimeSource) if runtimeSource.__class__ != Source else runtimeSource
        self.runtimeLine = runtimeLine
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        line = self.line
        endLine = self.endLine
        runtimeSource = self.runtimeSource
        runtimeLine = self.runtimeLine
        dct = {'line': line, 'endLine': endLine, 'runtimeSource': runtimeSource.to_dict(update_ids_to_dap=update_ids_to_dap), 'runtimeLine': runtimeLine}
        dct.update(self.kwargs)
        return dct