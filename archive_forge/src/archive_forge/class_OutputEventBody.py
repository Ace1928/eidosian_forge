from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class OutputEventBody(BaseSchema):
    """
    "body" of OutputEvent

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'category': {'type': 'string', 'description': "The output category. If not specified or if the category is not understand by the client, 'console' is assumed.", '_enum': ['console', 'important', 'stdout', 'stderr', 'telemetry'], 'enumDescriptions': ["Show the output in the client's default message UI, e.g. a 'debug console'. This category should only be used for informational output from the debugger (as opposed to the debuggee).", "A hint for the client to show the ouput in the client's UI for important and highly visible information, e.g. as a popup notification. This category should only be used for important messages from the debugger (as opposed to the debuggee). Since this category value is a hint, clients might ignore the hint and assume the 'console' category.", 'Show the output as normal program output from the debuggee.', 'Show the output as error program output from the debuggee.', 'Send the output to telemetry instead of showing it to the user.']}, 'output': {'type': 'string', 'description': 'The output to report.'}, 'group': {'type': 'string', 'description': 'Support for keeping an output log organized by grouping related messages.', 'enum': ['start', 'startCollapsed', 'end'], 'enumDescriptions': ["Start a new group in expanded mode. Subsequent output events are members of the group and should be shown indented.\nThe 'output' attribute becomes the name of the group and is not indented.", "Start a new group in collapsed mode. Subsequent output events are members of the group and should be shown indented (as soon as the group is expanded).\nThe 'output' attribute becomes the name of the group and is not indented.", "End the current group and decreases the indentation of subsequent output events.\nA non empty 'output' attribute is shown as the unindented end of the group."]}, 'variablesReference': {'type': 'integer', 'description': "If an attribute 'variablesReference' exists and its value is > 0, the output contains objects which can be retrieved by passing 'variablesReference' to the 'variables' request. The value should be less than or equal to 2147483647 (2^31-1)."}, 'source': {'description': 'An optional source location where the output was produced.', 'type': 'Source'}, 'line': {'type': 'integer', 'description': 'An optional source location line where the output was produced.'}, 'column': {'type': 'integer', 'description': 'An optional source location column where the output was produced.'}, 'data': {'type': ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'], 'description': "Optional data to report. For the 'telemetry' category the data will be sent to telemetry, for the other categories the data is shown in JSON format."}}
    __refs__ = set(['source'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, output, category=None, group=None, variablesReference=None, source=None, line=None, column=None, data=None, update_ids_from_dap=False, **kwargs):
        """
        :param string output: The output to report.
        :param string category: The output category. If not specified or if the category is not understand by the client, 'console' is assumed.
        :param string group: Support for keeping an output log organized by grouping related messages.
        :param integer variablesReference: If an attribute 'variablesReference' exists and its value is > 0, the output contains objects which can be retrieved by passing 'variablesReference' to the 'variables' request. The value should be less than or equal to 2147483647 (2^31-1).
        :param Source source: An optional source location where the output was produced.
        :param integer line: An optional source location line where the output was produced.
        :param integer column: An optional source location column where the output was produced.
        :param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] data: Optional data to report. For the 'telemetry' category the data will be sent to telemetry, for the other categories the data is shown in JSON format.
        """
        self.output = output
        self.category = category
        self.group = group
        self.variablesReference = variablesReference
        if source is None:
            self.source = Source()
        else:
            self.source = Source(update_ids_from_dap=update_ids_from_dap, **source) if source.__class__ != Source else source
        self.line = line
        self.column = column
        self.data = data
        if update_ids_from_dap:
            self.variablesReference = self._translate_id_from_dap(self.variablesReference)
        self.kwargs = kwargs

    @classmethod
    def update_dict_ids_from_dap(cls, dct):
        if 'variablesReference' in dct:
            dct['variablesReference'] = cls._translate_id_from_dap(dct['variablesReference'])
        return dct

    def to_dict(self, update_ids_to_dap=False):
        output = self.output
        category = self.category
        group = self.group
        variablesReference = self.variablesReference
        source = self.source
        line = self.line
        column = self.column
        data = self.data
        if update_ids_to_dap:
            if variablesReference is not None:
                variablesReference = self._translate_id_to_dap(variablesReference)
        dct = {'output': output}
        if category is not None:
            dct['category'] = category
        if group is not None:
            dct['group'] = group
        if variablesReference is not None:
            dct['variablesReference'] = variablesReference
        if source is not None:
            dct['source'] = source.to_dict(update_ids_to_dap=update_ids_to_dap)
        if line is not None:
            dct['line'] = line
        if column is not None:
            dct['column'] = column
        if data is not None:
            dct['data'] = data
        dct.update(self.kwargs)
        return dct

    @classmethod
    def update_dict_ids_to_dap(cls, dct):
        if 'variablesReference' in dct:
            dct['variablesReference'] = cls._translate_id_to_dap(dct['variablesReference'])
        return dct