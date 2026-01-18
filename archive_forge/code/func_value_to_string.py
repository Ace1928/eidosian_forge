import base64
import pickle
from django.db import models
from django.utils import encoding
import jsonpickle
import oauth2client
def value_to_string(self, obj):
    """Convert the field value from the provided model to a string.

        Used during model serialization.

        Args:
            obj: db.Model, model object

        Returns:
            string, the serialized field value
        """
    value = self._get_val_from_obj(obj)
    return self.get_prep_value(value)