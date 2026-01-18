import hashlib
import json
from oslo_utils import encodeutils
from requests import codes
import urllib.parse
import warlock
from glanceclient.common import utils
from glanceclient import exc
from glanceclient.v2 import schemas
@utils.memoized_property
def unvalidated_model(self):
    """A model which does not validate the image against the v2 schema."""
    schema = self.schema_client.get('image')
    warlock_model = warlock.model_factory(schema.raw(), base_class=schemas.SchemaBasedModel)
    warlock_model.validate = lambda *args, **kwargs: None
    return warlock_model