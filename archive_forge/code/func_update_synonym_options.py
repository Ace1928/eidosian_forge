import boto
import boto.jsonresponse
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def update_synonym_options(self, domain_name, synonyms):
    """
        Updates synonym options used by indexing for the search domain.

        :type domain_name: string
        :param domain_name: A string that represents the name of a
            domain. Domain names must be unique across the domains
            owned by an account within an AWS region. Domain names
            must start with a letter or number and can contain the
            following characters: a-z (lowercase), 0-9, and -
            (hyphen). Uppercase letters and underscores are not
            allowed.

        :type synonyms: string
        :param synonyms: Maps terms to their synonyms.  The JSON object
            has a single key "synonyms" whose value is a dict mapping terms
            to their synonyms. Each synonym is a simple string or an
            array of strings. The maximum size of a stopwords document
            is 100KB. Example:
            {"synonyms": {"cat": ["feline", "kitten"], "puppy": "dog"}}

        :raises: BaseException, InternalException, InvalidTypeException,
            LimitExceededException, ResourceNotFoundException
        """
    doc_path = ('update_synonym_options_response', 'update_synonym_options_result', 'synonyms')
    params = {'DomainName': domain_name, 'Synonyms': synonyms}
    return self.get_response(doc_path, 'UpdateSynonymOptions', params, verb='POST')