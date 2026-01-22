import boto.exception
from boto.compat import json
import requests
import boto
def _check_num_ops(self, type_, response_num):
    """Raise exception if number of ops in response doesn't match commit

        :type type_: str
        :param type_: Type of commit operation: 'add' or 'delete'

        :type response_num: int
        :param response_num: Number of adds or deletes in the response.

        :raises: :class:`boto.cloudsearch.document.CommitMismatchError`
        """
    commit_num = len([d for d in self.doc_service.documents_batch if d['type'] == type_])
    if response_num != commit_num:
        raise CommitMismatchError('Incorrect number of {0}s returned. Commit: {1} Response: {2}'.format(type_, commit_num, response_num))