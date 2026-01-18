import html
from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...utils import is_bs4_available, logging, requires_backends
def xpath_soup(self, element):
    xpath_tags = []
    xpath_subscripts = []
    child = element if element.name else element.parent
    for parent in child.parents:
        siblings = parent.find_all(child.name, recursive=False)
        xpath_tags.append(child.name)
        xpath_subscripts.append(0 if 1 == len(siblings) else next((i for i, s in enumerate(siblings, 1) if s is child)))
        child = parent
    xpath_tags.reverse()
    xpath_subscripts.reverse()
    return (xpath_tags, xpath_subscripts)