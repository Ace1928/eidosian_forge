import io
from oslo_log.cmds import convert_json
from oslo_serialization import jsonutils
from oslotest import base as test_base
def test_reformat_json_single(self):
    text = jsonutils.dumps(TRIVIAL_RECORD)
    self.assertEqual([TRIVIAL_RECORD], self._reformat(text))