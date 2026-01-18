from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.cloudsearchdomain import exceptions
def upload_documents(self, documents, content_type):
    """
        Posts a batch of documents to a search domain for indexing. A
        document batch is a collection of add and delete operations
        that represent the documents you want to add, update, or
        delete from your domain. Batches can be described in either
        JSON or XML. Each item that you want Amazon CloudSearch to
        return as a search result (such as a product) is represented
        as a document. Every document has a unique ID and one or more
        fields that contain the data that you want to search and
        return in results. Individual documents cannot contain more
        than 1 MB of data. The entire batch cannot exceed 5 MB. To get
        the best possible upload performance, group add and delete
        operations in batches that are close the 5 MB limit.
        Submitting a large volume of single-document batches can
        overload a domain's document service.

        The endpoint for submitting `UploadDocuments` requests is
        domain-specific. To get the document endpoint for your domain,
        use the Amazon CloudSearch configuration service
        `DescribeDomains` action. A domain's endpoints are also
        displayed on the domain dashboard in the Amazon CloudSearch
        console.

        For more information about formatting your data for Amazon
        CloudSearch, see `Preparing Your Data`_ in the Amazon
        CloudSearch Developer Guide . For more information about
        uploading data for indexing, see `Uploading Data`_ in the
        Amazon CloudSearch Developer Guide .

        :type documents: blob
        :param documents: A batch of documents formatted in JSON or HTML.

        :type content_type: string
        :param content_type:
        The format of the batch you are uploading. Amazon CloudSearch supports
            two document batch formats:


        + application/json
        + application/xml

        """
    uri = '/2013-01-01/documents/batch'
    headers = {}
    query_params = {}
    if content_type is not None:
        headers['Content-Type'] = content_type
    return self.make_request('POST', uri, expected_status=200, data=documents, headers=headers, params=query_params)