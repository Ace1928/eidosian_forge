from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _handle_update_tags(self, prop_diff):
    self.my_image.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.image_tags.update.assert_called_once_with(self.my_image.resource_id, 'tag2')
    self.image_tags.delete.assert_called_once_with(self.my_image.resource_id, 'tag1')