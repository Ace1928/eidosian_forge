from typing import Any, Dict, List, Optional
from tune.concepts.space import to_template
from tune.concepts.space.parameters import TuningParametersTemplate
def with_dfs(self, dfs: Dict[str, Any]) -> 'Trial':
    """Set dataframes for the trial, a new Trial object will
        be constructed and with the new ``dfs``

        :param dfs: dataframes to attach to the trial

        """
    if len(dfs) == 0 and len(self.dfs) == 0:
        return self
    t = self.copy()
    t._dfs = dfs
    return t