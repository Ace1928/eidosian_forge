import boto
from boto.compat import json
from boto.cloudsearch.optionstatus import OptionStatus
from boto.cloudsearch.optionstatus import IndexFieldStatus
from boto.cloudsearch.optionstatus import ServicePoliciesStatus
from boto.cloudsearch.optionstatus import RankExpressionStatus
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.search import SearchConnection
def update_from_data(self, data):
    self.created = data['created']
    self.deleted = data['deleted']
    self.processing = data['processing']
    self.requires_index_documents = data['requires_index_documents']
    self.domain_id = data['domain_id']
    self.domain_name = data['domain_name']
    self.num_searchable_docs = data['num_searchable_docs']
    self.search_instance_count = data['search_instance_count']
    self.search_instance_type = data.get('search_instance_type', None)
    self.search_partition_count = data['search_partition_count']
    self._doc_service = data['doc_service']
    self._search_service = data['search_service']