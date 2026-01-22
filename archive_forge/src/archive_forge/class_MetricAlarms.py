from datetime import datetime
from boto.ec2.cloudwatch.listelement import ListElement
from boto.ec2.cloudwatch.dimension import Dimension
from boto.compat import json
from boto.compat import six
class MetricAlarms(list):

    def __init__(self, connection=None):
        """
        Parses a list of MetricAlarms.
        """
        list.__init__(self)
        self.connection = connection

    def startElement(self, name, attrs, connection):
        if name == 'member':
            metric_alarm = MetricAlarm(connection)
            self.append(metric_alarm)
            return metric_alarm

    def endElement(self, name, value, connection):
        pass