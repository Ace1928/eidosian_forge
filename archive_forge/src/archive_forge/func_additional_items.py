from pprint import pformat
from six import iteritems
import re
@additional_items.setter
def additional_items(self, additional_items):
    """
        Sets the additional_items of this V1beta1JSONSchemaProps.
        JSONSchemaPropsOrBool represents JSONSchemaProps or a boolean value.
        Defaults to true for the boolean property.

        :param additional_items: The additional_items of this
        V1beta1JSONSchemaProps.
        :type: object
        """
    self._additional_items = additional_items