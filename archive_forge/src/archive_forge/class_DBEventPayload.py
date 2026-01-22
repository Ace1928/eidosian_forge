class DBEventPayload(EventPayload):
    """The payload for data store events payloads."""

    def __init__(self, context, metadata=None, request_body=None, states=None, resource_id=None, desired_state=None):
        super().__init__(context, metadata=metadata, request_body=request_body, states=states, resource_id=resource_id)
        self.desired_state = desired_state

    @property
    def is_persisted(self):
        """Determine if the resource for this event payload is persisted.

        :returns: True if this payload's resource is persisted, otherwise
            False.
        """
        return self.resource_id is not None and self.has_states

    @property
    def is_to_be_committed(self):
        """"Determine if the event payload resource is to be committed.

        :returns:  True if the desired state has been populated, else False.
        """
        return self.desired_state is not None

    @property
    def latest_state(self):
        """Returns the latest state for the event payload resource.

        :returns: If this payload has a desired_state its returned, otherwise
            latest_state is returned.
        """
        return self.desired_state or super().latest_state