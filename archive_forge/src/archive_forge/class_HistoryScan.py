import json
import requests
from wandb_gql import gql
from wandb_gql.client import RetryError
from wandb import util
from wandb.apis.normalize import normalize_exceptions
from wandb.sdk.lib import retry
class HistoryScan:
    QUERY = gql('\n        query HistoryPage($entity: String!, $project: String!, $run: String!, $minStep: Int64!, $maxStep: Int64!, $pageSize: Int!) {\n            project(name: $project, entityName: $entity) {\n                run(name: $run) {\n                    history(minStep: $minStep, maxStep: $maxStep, samples: $pageSize)\n                }\n            }\n        }\n        ')

    def __init__(self, client, run, min_step, max_step, page_size=1000):
        self.client = client
        self.run = run
        self.page_size = page_size
        self.min_step = min_step
        self.max_step = max_step
        self.page_offset = min_step
        self.scan_offset = 0
        self.rows = []

    def __iter__(self):
        self.page_offset = self.min_step
        self.scan_offset = 0
        self.rows = []
        return self

    def __next__(self):
        while True:
            if self.scan_offset < len(self.rows):
                row = self.rows[self.scan_offset]
                self.scan_offset += 1
                return row
            if self.page_offset >= self.max_step:
                raise StopIteration()
            self._load_next()
    next = __next__

    @normalize_exceptions
    @retry.retriable(check_retry_fn=util.no_retry_auth, retryable_exceptions=(RetryError, requests.RequestException))
    def _load_next(self):
        max_step = self.page_offset + self.page_size
        if max_step > self.max_step:
            max_step = self.max_step
        variables = {'entity': self.run.entity, 'project': self.run.project, 'run': self.run.id, 'minStep': int(self.page_offset), 'maxStep': int(max_step), 'pageSize': int(self.page_size)}
        res = self.client.execute(self.QUERY, variable_values=variables)
        res = res['project']['run']['history']
        self.rows = [json.loads(row) for row in res]
        self.page_offset += self.page_size
        self.scan_offset = 0