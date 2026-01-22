from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def _to_tags(self, data):
    """
        return tags dict
        """
    tags = {}
    xpath = 'DescribeTagsResult/TagDescriptions/member/Tags/member'
    for el in findall(element=data, xpath=xpath, namespace=NS):
        key = findtext(element=el, xpath='Key', namespace=NS)
        value = findtext(element=el, xpath='Value', namespace=NS)
        if key:
            tags[key] = value
    return tags