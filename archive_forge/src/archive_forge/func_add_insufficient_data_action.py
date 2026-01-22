from datetime import datetime
from boto.ec2.cloudwatch.listelement import ListElement
from boto.ec2.cloudwatch.dimension import Dimension
from boto.compat import json
from boto.compat import six
def add_insufficient_data_action(self, action_arn=None):
    """
        Adds an insufficient_data action, represented as an SNS topic, to
        this alarm. What to do when the insufficient_data state is reached.

        :type action_arn: str
        :param action_arn: SNS topics to which notification should be
                           sent if the alarm goes to state INSUFFICIENT_DATA.
        """
    if not action_arn:
        return
    self.actions_enabled = 'true'
    self.insufficient_data_actions.append(action_arn)