import json
import logging
import sys
from threading import RLock
from typing import Any, Dict, Optional
import requests
from ray.autoscaler.node_launch_exception import NodeLaunchException
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def _query_node_status(self, node_id):
    spark_job_group_id = self._gen_spark_job_group_id(node_id)
    response = requests.post(url=self.spark_job_server_url + '/query_task_status', json={'spark_job_group_id': spark_job_group_id})
    response.raise_for_status()
    decoded_resp = response.content.decode('utf-8')
    json_res = json.loads(decoded_resp)
    return json_res['status']