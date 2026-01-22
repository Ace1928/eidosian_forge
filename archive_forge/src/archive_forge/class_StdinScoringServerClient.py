import json
import logging
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import requests
from mlflow.deployments import PredictionsResponse
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import scoring_server
from mlflow.utils.proto_json_utils import dump_input_data
class StdinScoringServerClient(BaseScoringServerClient):

    def __init__(self, process):
        super().__init__()
        self.process = process
        self.tmpdir = Path(tempfile.mkdtemp())
        self.output_json = self.tmpdir.joinpath('output.json')

    def wait_server_ready(self, timeout=30, scoring_server_proc=None):
        return_code = self.process.poll()
        if return_code is not None:
            raise RuntimeError(f'Server process already exit with returncode {return_code}')

    def invoke(self, data, params: Optional[Dict[str, Any]]=None):
        """
        Invoke inference on input data. The input data must be pandas dataframe or numpy array or
        a dict of numpy arrays.

        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                           release without warning.

        Returns:
            :py:class:`PredictionsResponse <mlflow.deployments.PredictionsResponse>` result.
        """
        if not self.output_json.exists():
            self.output_json.touch()
        request_id = str(uuid.uuid4())
        request = {'id': request_id, 'data': dump_input_data(data, params=params), 'output_file': str(self.output_json)}
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        begin_time = time.time()
        while True:
            _logger.info('Waiting for scoring to complete...')
            try:
                with self.output_json.open(mode='r+') as f:
                    resp = PredictionsResponse.from_json(f.read())
                    if resp.get('id') == request_id:
                        f.truncate(0)
                        return resp
            except Exception as e:
                _logger.debug('Exception while waiting for scoring to complete: %s', e)
            if time.time() - begin_time > 60:
                raise MlflowException('Scoring timeout')
            time.sleep(1)