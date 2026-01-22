from typing import Any, Dict, List, Union
from utils.log import quick_log
from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel
import gc
import copy
import global_var
def __get_a_dtrie_buff_size(dtrie_v):
    return 54 * len(dtrie_v['tokens']) + 491520 + 262144 + 28