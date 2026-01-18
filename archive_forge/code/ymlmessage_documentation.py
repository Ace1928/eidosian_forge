from boto.sqs.message import Message
import yaml

    The YAMLMessage class provides a YAML compatible message. Encoding and
    decoding are handled automaticaly.

    Access this message data like such:

    m.data = [ 1, 2, 3]
    m.data[0] # Returns 1

    This depends on the PyYAML package
    