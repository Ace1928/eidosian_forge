import csv
import datetime
import json
import logging
import apache_beam as beam
from six.moves import cStringIO
import yaml
from google.cloud.ml.util import _decoders
from google.cloud.ml.util import _file
class TrainingJobResult(object):
    """Result of training a model."""

    def __init__(self):
        self.training_request = None
        self.training_job_metadata = None
        self.error = None
        self.training_job_result = None

    def __eq__(self, o):
        for f in ['training_request', 'training_job_metadata', 'error', 'training_job_result']:
            if getattr(self, f) != getattr(o, f):
                return False
        return True

    def __ne__(self, o):
        return not self == o

    def __repr__(self):
        fields = []
        for k, v in self.__dict__.iteritems():
            fields.append('{0}={1}'.format(k, v))
        return 'TrainingJobResult({0})'.format(', '.join(fields))