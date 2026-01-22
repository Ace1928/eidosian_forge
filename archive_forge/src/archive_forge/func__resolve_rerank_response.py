import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from wandb.sdk.lib.runid import generate_id
def _resolve_rerank_response(self, response: Response) -> List[Dict[str, Any]]:
    flattened_response_dicts = [flatten_dict(_response.__dict__) for _response in response]
    return_dict = collect_common_keys(flattened_response_dicts)
    return_dict['id'] = response.id
    return [return_dict]