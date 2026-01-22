class EventPayload(object):
    """Base event payload object.

    This class is intended to be the super class for all event payloads. As
    such, it defines common attributes many events are likely to use in their
    payload. Note that event attributes are passed by reference; no copying
    of states, metadata or request_body is performed and thus consumers should
    not modify payload references.

    For more information, see the callbacks dev-ref documentation for this
    project.
    """

    def __init__(self, context, metadata=None, request_body=None, states=None, resource_id=None):
        self.context = context
        self.metadata = metadata if metadata else {}
        self.request_body = request_body
        self.states = states if states else []
        self.resource_id = resource_id

    @property
    def has_states(self):
        """Determines if this event payload has any states.

        :returns: True if this event payload has states, otherwise False.
        """
        return len(self.states) > 0

    @property
    def latest_state(self):
        """Returns the latest state for the event payload.

        :returns: The last state of this event payload if has_state else None.
        """
        return self.states[-1] if self.has_states else None