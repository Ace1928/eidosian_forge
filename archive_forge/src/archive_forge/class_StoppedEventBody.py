from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class StoppedEventBody(BaseSchema):
    """
    "body" of StoppedEvent

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'reason': {'type': 'string', 'description': "The reason for the event.\nFor backward compatibility this string is shown in the UI if the 'description' attribute is missing (but it must not be translated).", '_enum': ['step', 'breakpoint', 'exception', 'pause', 'entry', 'goto', 'function breakpoint', 'data breakpoint', 'instruction breakpoint']}, 'description': {'type': 'string', 'description': "The full reason for the event, e.g. 'Paused on exception'. This string is shown in the UI as is and must be translated."}, 'threadId': {'type': 'integer', 'description': 'The thread which was stopped.'}, 'preserveFocusHint': {'type': 'boolean', 'description': 'A value of true hints to the frontend that this event should not change the focus.'}, 'text': {'type': 'string', 'description': "Additional information. E.g. if reason is 'exception', text contains the exception name. This string is shown in the UI."}, 'allThreadsStopped': {'type': 'boolean', 'description': "If 'allThreadsStopped' is true, a debug adapter can announce that all threads have stopped.\n- The client should use this information to enable that all threads can be expanded to access their stacktraces.\n- If the attribute is missing or false, only the thread with the given threadId can be expanded."}, 'hitBreakpointIds': {'type': 'array', 'items': {'type': 'integer'}, 'description': 'Ids of the breakpoints that triggered the event. In most cases there will be only a single breakpoint but here are some examples for multiple breakpoints:\n- Different types of breakpoints map to the same location.\n- Multiple source breakpoints get collapsed to the same instruction by the compiler/runtime.\n- Multiple function breakpoints with different function names map to the same location.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, reason, description=None, threadId=None, preserveFocusHint=None, text=None, allThreadsStopped=None, hitBreakpointIds=None, update_ids_from_dap=False, **kwargs):
        """
        :param string reason: The reason for the event.
        For backward compatibility this string is shown in the UI if the 'description' attribute is missing (but it must not be translated).
        :param string description: The full reason for the event, e.g. 'Paused on exception'. This string is shown in the UI as is and must be translated.
        :param integer threadId: The thread which was stopped.
        :param boolean preserveFocusHint: A value of true hints to the frontend that this event should not change the focus.
        :param string text: Additional information. E.g. if reason is 'exception', text contains the exception name. This string is shown in the UI.
        :param boolean allThreadsStopped: If 'allThreadsStopped' is true, a debug adapter can announce that all threads have stopped.
        - The client should use this information to enable that all threads can be expanded to access their stacktraces.
        - If the attribute is missing or false, only the thread with the given threadId can be expanded.
        :param array hitBreakpointIds: Ids of the breakpoints that triggered the event. In most cases there will be only a single breakpoint but here are some examples for multiple breakpoints:
        - Different types of breakpoints map to the same location.
        - Multiple source breakpoints get collapsed to the same instruction by the compiler/runtime.
        - Multiple function breakpoints with different function names map to the same location.
        """
        self.reason = reason
        self.description = description
        self.threadId = threadId
        self.preserveFocusHint = preserveFocusHint
        self.text = text
        self.allThreadsStopped = allThreadsStopped
        self.hitBreakpointIds = hitBreakpointIds
        if update_ids_from_dap:
            self.threadId = self._translate_id_from_dap(self.threadId)
        self.kwargs = kwargs

    @classmethod
    def update_dict_ids_from_dap(cls, dct):
        if 'threadId' in dct:
            dct['threadId'] = cls._translate_id_from_dap(dct['threadId'])
        return dct

    def to_dict(self, update_ids_to_dap=False):
        reason = self.reason
        description = self.description
        threadId = self.threadId
        preserveFocusHint = self.preserveFocusHint
        text = self.text
        allThreadsStopped = self.allThreadsStopped
        hitBreakpointIds = self.hitBreakpointIds
        if hitBreakpointIds and hasattr(hitBreakpointIds[0], 'to_dict'):
            hitBreakpointIds = [x.to_dict() for x in hitBreakpointIds]
        if update_ids_to_dap:
            if threadId is not None:
                threadId = self._translate_id_to_dap(threadId)
        dct = {'reason': reason}
        if description is not None:
            dct['description'] = description
        if threadId is not None:
            dct['threadId'] = threadId
        if preserveFocusHint is not None:
            dct['preserveFocusHint'] = preserveFocusHint
        if text is not None:
            dct['text'] = text
        if allThreadsStopped is not None:
            dct['allThreadsStopped'] = allThreadsStopped
        if hitBreakpointIds is not None:
            dct['hitBreakpointIds'] = hitBreakpointIds
        dct.update(self.kwargs)
        return dct

    @classmethod
    def update_dict_ids_to_dap(cls, dct):
        if 'threadId' in dct:
            dct['threadId'] = cls._translate_id_to_dap(dct['threadId'])
        return dct