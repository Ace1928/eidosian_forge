import csv
import datetime
import json
import logging
import apache_beam as beam
from six.moves import cStringIO
import yaml
from google.cloud.ml.util import _decoders
from google.cloud.ml.util import _file
class TrainingJobResultCoder(beam.coders.Coder):
    """Custom coder for TrainingJobResult."""

    def encode(self, training_job_result):
        """Encode a TrainingJobResult object into a JSON string.

    Args:
      training_job_result: A TrainingJobResult object.

    Returns:
      A JSON string
    """
        d = {}
        d.update(training_job_result.__dict__)
        if d['training_request'] is not None:
            coder = TrainingJobRequestCoder()
            d['training_request'] = coder.encode(d['training_request'])
        return json.dumps(d)

    def decode(self, training_job_result_string):
        """Decode a string to a TrainingJobResult object.

    Args:
      training_job_result_string: A string representing a TrainingJobResult.

    Returns:
      A TrainingJobResult object.
    """
        r = TrainingJobResult()
        d = json.loads(training_job_result_string)
        if d['training_request'] is not None:
            coder = TrainingJobRequestCoder()
            d['training_request'] = coder.decode(d['training_request'])
        r.__dict__.update(d)
        return r