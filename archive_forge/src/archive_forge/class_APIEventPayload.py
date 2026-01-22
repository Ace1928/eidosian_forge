class APIEventPayload(EventPayload):
    """The payload for API events."""

    def __init__(self, context, method_name, action, metadata=None, request_body=None, states=None, resource_id=None, collection_name=None):
        super().__init__(context, metadata=metadata, request_body=request_body, states=states, resource_id=resource_id)
        self.method_name = method_name
        self.action = action
        self.collection_name = collection_name