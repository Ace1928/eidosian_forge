from collections import deque
from http.server import HTTPServer, SimpleHTTPRequestHandler
import logging
import queue
from socketserver import ThreadingMixIn
import threading
import time
import traceback
from typing import List
import ray.cloudpickle as pickle
from ray.rllib.env.policy_client import (
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.evaluation.sampler import SamplerInput
from ray.rllib.utils.typing import SampleBatchType
def _make_handler(rollout_worker, samples_queue, metrics_queue):
    child_rollout_worker = None
    inference_thread = None
    lock = threading.Lock()

    def setup_child_rollout_worker():
        nonlocal lock
        with lock:
            nonlocal child_rollout_worker
            nonlocal inference_thread
            if child_rollout_worker is None:
                child_rollout_worker, inference_thread = _create_embedded_rollout_worker(rollout_worker.creation_args(), report_data)
                child_rollout_worker.set_weights(rollout_worker.get_weights())

    def report_data(data):
        nonlocal child_rollout_worker
        batch = data['samples']
        batch.decompress_if_needed()
        samples_queue.append(batch)
        if len(samples_queue) == samples_queue.maxlen:
            logger.warning('PolicyServerInput queue is full! Purging half of the samples (oldest).')
            for _ in range(samples_queue.maxlen // 2):
                samples_queue.popleft()
        for rollout_metric in data['metrics']:
            metrics_queue.put(rollout_metric)
        if child_rollout_worker is not None:
            child_rollout_worker.set_weights(rollout_worker.get_weights(), rollout_worker.get_global_vars())

    class Handler(SimpleHTTPRequestHandler):

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)

        def do_POST(self):
            content_len = int(self.headers.get('Content-Length'), 0)
            raw_body = self.rfile.read(content_len)
            parsed_input = pickle.loads(raw_body)
            try:
                response = self.execute_command(parsed_input)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(pickle.dumps(response))
            except Exception:
                self.send_error(500, traceback.format_exc())

        def execute_command(self, args):
            command = args['command']
            response = {}
            if command == Commands.GET_WORKER_ARGS:
                logger.info('Sending worker creation args to client.')
                response['worker_args'] = rollout_worker.creation_args()
            elif command == Commands.GET_WEIGHTS:
                logger.info('Sending worker weights to client.')
                response['weights'] = rollout_worker.get_weights()
                response['global_vars'] = rollout_worker.get_global_vars()
            elif command == Commands.REPORT_SAMPLES:
                logger.info('Got sample batch of size {} from client.'.format(args['samples'].count))
                report_data(args)
            elif command == Commands.START_EPISODE:
                setup_child_rollout_worker()
                assert inference_thread.is_alive()
                response['episode_id'] = child_rollout_worker.env.start_episode(args['episode_id'], args['training_enabled'])
            elif command == Commands.GET_ACTION:
                assert inference_thread.is_alive()
                response['action'] = child_rollout_worker.env.get_action(args['episode_id'], args['observation'])
            elif command == Commands.LOG_ACTION:
                assert inference_thread.is_alive()
                child_rollout_worker.env.log_action(args['episode_id'], args['observation'], args['action'])
            elif command == Commands.LOG_RETURNS:
                assert inference_thread.is_alive()
                if args['done']:
                    child_rollout_worker.env.log_returns(args['episode_id'], args['reward'], args['info'], args['done'])
                else:
                    child_rollout_worker.env.log_returns(args['episode_id'], args['reward'], args['info'])
            elif command == Commands.END_EPISODE:
                assert inference_thread.is_alive()
                child_rollout_worker.env.end_episode(args['episode_id'], args['observation'])
            else:
                raise ValueError('Unknown command: {}'.format(command))
            return response
    return Handler