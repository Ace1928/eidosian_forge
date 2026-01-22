from collections import Counter
from threading import Timer
import logging
import inspect
from ..core import MachineError, listify, State
class Error(Tags):
    """ This mix in builds upon tag and should be used INSTEAD of Tags if final states that have
        not been tagged with 'accepted' should throw an `MachineError`.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            **kwargs: If kwargs contains the keyword `accepted` add the 'accepted' tag to a tag list
                which will be forwarded to the Tags constructor.
        """
        tags = kwargs.get('tags', [])
        accepted = kwargs.pop('accepted', False)
        if accepted:
            tags.append('accepted')
            kwargs['tags'] = tags
        super(Error, self).__init__(*args, **kwargs)

    def enter(self, event_data):
        """ Extends transitions.core.State.enter. Throws a `MachineError` if there is
            no leaving transition from this state and 'accepted' is not in self.tags.
        """
        if not event_data.machine.get_triggers(self.name) and (not self.is_accepted):
            raise MachineError("Error state '{0}' reached!".format(self.name))
        super(Error, self).enter(event_data)