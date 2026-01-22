from __future__ import absolute_import
import copy
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
def _HandleEvents(self, events):
    """Iterate over all events and send them to handler.

    This method is not meant to be called from the interface.

    Only use in tests.

    Args:
      events: Iterator or generator containing events to process.
    raises:
      EventListenerParserError when a yaml.parser.ParserError is raised.
      EventError when an exception occurs during the handling of an event.
    """
    for event in events:
        try:
            self.HandleEvent(*event)
        except Exception as e:
            event_object, loader = event
            raise yaml_errors.EventError(e, event_object)