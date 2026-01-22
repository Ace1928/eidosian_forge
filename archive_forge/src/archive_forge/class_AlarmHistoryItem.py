from datetime import datetime
from boto.ec2.cloudwatch.listelement import ListElement
from boto.ec2.cloudwatch.dimension import Dimension
from boto.compat import json
from boto.compat import six
class AlarmHistoryItem(object):

    def __init__(self, connection=None):
        self.connection = connection

    def __repr__(self):
        return 'AlarmHistory:%s[%s at %s]' % (self.name, self.summary, self.timestamp)

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'AlarmName':
            self.name = value
        elif name == 'HistoryData':
            self.data = json.loads(value)
        elif name == 'HistoryItemType':
            self.tem_type = value
        elif name == 'HistorySummary':
            self.summary = value
        elif name == 'Timestamp':
            try:
                self.timestamp = datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%fZ')
            except ValueError:
                self.timestamp = datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')