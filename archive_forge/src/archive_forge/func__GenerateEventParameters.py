from __future__ import absolute_import
import copy
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
def _GenerateEventParameters(self, stream, loader_class=yaml.loader.SafeLoader, **loader_args):
    """Creates a generator that yields event, loader parameter pairs.

    For use as parameters to HandleEvent method for use by Parse method.
    During testing, _GenerateEventParameters is simulated by allowing
    the harness to pass in a list of pairs as the parameter.

    A list of (event, loader) pairs must be passed to _HandleEvents otherwise
    it is not possible to pass the loader instance to the handler.

    Also responsible for instantiating the loader from the Loader
    parameter.

    Args:
      stream: String document or open file object to process as per the
        yaml.parse method.  Any object that implements a 'read()' method which
        returns a string document will work.
      loader_class: Loader class to use as per the yaml.parse method.  Used to
        instantiate new yaml.loader instance.
      **loader_args: Pass to the loader on construction


    Yields:
      Tuple(event, loader) where:
        event: Event emitted by PyYAML loader.
        loader: Used for dependency injection.
    """
    assert loader_class is not None
    try:
        loader = loader_class(stream, **loader_args)
        while loader.check_event():
            yield (loader.get_event(), loader)
    except yaml.error.YAMLError as e:
        raise yaml_errors.EventListenerYAMLError(e)