from typing import TYPE_CHECKING, Any, Dict
from docutils.transforms.references import DanglingReferences
from sphinx.transforms import SphinxTransform
class SphinxDomains(SphinxTransform):
    """Collect objects to Sphinx domains for cross references."""
    default_priority = 850

    def apply(self, **kwargs: Any) -> None:
        for domain in self.env.domains.values():
            domain.process_doc(self.env, self.env.docname, self.document)