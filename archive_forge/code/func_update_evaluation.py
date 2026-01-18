import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def update_evaluation(self, evaluation_id, evaluation_name):
    """
        Updates the `EvaluationName` of an `Evaluation`.

        You can use the GetEvaluation operation to view the contents
        of the updated data element.

        :type evaluation_id: string
        :param evaluation_id: The ID assigned to the `Evaluation` during
            creation.

        :type evaluation_name: string
        :param evaluation_name: A new user-supplied name or description of the
            `Evaluation` that will replace the current content.

        """
    params = {'EvaluationId': evaluation_id, 'EvaluationName': evaluation_name}
    return self.make_request(action='UpdateEvaluation', body=json.dumps(params))