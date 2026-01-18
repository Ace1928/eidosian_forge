import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def update_ml_model(self, ml_model_id, ml_model_name=None, score_threshold=None):
    """
        Updates the `MLModelName` and the `ScoreThreshold` of an
        `MLModel`.

        You can use the GetMLModel operation to view the contents of
        the updated data element.

        :type ml_model_id: string
        :param ml_model_id: The ID assigned to the `MLModel` during creation.

        :type ml_model_name: string
        :param ml_model_name: A user-supplied name or description of the
            `MLModel`.

        :type score_threshold: float
        :param score_threshold: The `ScoreThreshold` used in binary
            classification `MLModel` that marks the boundary between a positive
            prediction and a negative prediction.
        Output values greater than or equal to the `ScoreThreshold` receive a
            positive result from the `MLModel`, such as `True`. Output values
            less than the `ScoreThreshold` receive a negative response from the
            `MLModel`, such as `False`.

        """
    params = {'MLModelId': ml_model_id}
    if ml_model_name is not None:
        params['MLModelName'] = ml_model_name
    if score_threshold is not None:
        params['ScoreThreshold'] = score_threshold
    return self.make_request(action='UpdateMLModel', body=json.dumps(params))