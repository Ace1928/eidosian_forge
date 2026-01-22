import cffi  # type: ignore
import os
import re
import sys
import warnings
import situation  # preloaded in setup.py
import importlib
def _c_preprocess_block(csource, definitions={}, rownum=0):
    localdefs = definitions.copy()
    block = []
    for row in csource:
        rownum += 1
        m = DEFINE_PAT.match(row)
        if m:
            localdefs[m.group(1)] = m.group(2)
            continue
        m_ifdef = IFDEF_PAT.match(row)
        if m_ifdef:
            subblock, subdefs = _c_preprocess_ifdef(csource, m_ifdef.group(1) in localdefs, definitions=localdefs, rownum=rownum)
            block.extend(subblock)
            definitions.update(subdefs)
            continue
        m_else = ELSE_PAT.match(row)
        if m_else:
            return ('else', block, definitions)
        m_endif = ENDIF_PAT.match(row)
        if m_endif:
            return ('endif', block, definitions)
        for k, v in localdefs.items():
            if isinstance(v, str):
                row = row.replace(k, v)
        block.append(row)