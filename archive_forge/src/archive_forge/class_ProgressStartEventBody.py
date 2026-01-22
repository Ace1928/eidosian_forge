from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ProgressStartEventBody(BaseSchema):
    """
    "body" of ProgressStartEvent

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'progressId': {'type': 'string', 'description': "An ID that must be used in subsequent 'progressUpdate' and 'progressEnd' events to make them refer to the same progress reporting.\nIDs must be unique within a debug session."}, 'title': {'type': 'string', 'description': 'Mandatory (short) title of the progress reporting. Shown in the UI to describe the long running operation.'}, 'requestId': {'type': 'integer', 'description': 'The request ID that this progress report is related to. If specified a debug adapter is expected to emit\nprogress events for the long running request until the request has been either completed or cancelled.\nIf the request ID is omitted, the progress report is assumed to be related to some general activity of the debug adapter.'}, 'cancellable': {'type': 'boolean', 'description': "If true, the request that reports progress may be canceled with a 'cancel' request.\nSo this property basically controls whether the client should use UX that supports cancellation.\nClients that don't support cancellation are allowed to ignore the setting."}, 'message': {'type': 'string', 'description': 'Optional, more detailed progress message.'}, 'percentage': {'type': 'number', 'description': 'Optional progress percentage to display (value range: 0 to 100). If omitted no percentage will be shown.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, progressId, title, requestId=None, cancellable=None, message=None, percentage=None, update_ids_from_dap=False, **kwargs):
        """
        :param string progressId: An ID that must be used in subsequent 'progressUpdate' and 'progressEnd' events to make them refer to the same progress reporting.
        IDs must be unique within a debug session.
        :param string title: Mandatory (short) title of the progress reporting. Shown in the UI to describe the long running operation.
        :param integer requestId: The request ID that this progress report is related to. If specified a debug adapter is expected to emit
        progress events for the long running request until the request has been either completed or cancelled.
        If the request ID is omitted, the progress report is assumed to be related to some general activity of the debug adapter.
        :param boolean cancellable: If true, the request that reports progress may be canceled with a 'cancel' request.
        So this property basically controls whether the client should use UX that supports cancellation.
        Clients that don't support cancellation are allowed to ignore the setting.
        :param string message: Optional, more detailed progress message.
        :param number percentage: Optional progress percentage to display (value range: 0 to 100). If omitted no percentage will be shown.
        """
        self.progressId = progressId
        self.title = title
        self.requestId = requestId
        self.cancellable = cancellable
        self.message = message
        self.percentage = percentage
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        progressId = self.progressId
        title = self.title
        requestId = self.requestId
        cancellable = self.cancellable
        message = self.message
        percentage = self.percentage
        dct = {'progressId': progressId, 'title': title}
        if requestId is not None:
            dct['requestId'] = requestId
        if cancellable is not None:
            dct['cancellable'] = cancellable
        if message is not None:
            dct['message'] = message
        if percentage is not None:
            dct['percentage'] = percentage
        dct.update(self.kwargs)
        return dct