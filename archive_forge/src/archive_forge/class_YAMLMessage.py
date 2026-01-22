from boto.sqs.message import Message
import yaml
class YAMLMessage(Message):
    """
    The YAMLMessage class provides a YAML compatible message. Encoding and
    decoding are handled automaticaly.

    Access this message data like such:

    m.data = [ 1, 2, 3]
    m.data[0] # Returns 1

    This depends on the PyYAML package
    """

    def __init__(self, queue=None, body='', xml_attrs=None):
        self.data = None
        super(YAMLMessage, self).__init__(queue, body)

    def set_body(self, body):
        self.data = yaml.safe_load(body)

    def get_body(self):
        return yaml.dump(self.data)