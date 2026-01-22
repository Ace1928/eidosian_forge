from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
import re
import traceback
def add_vdooptions(params):
    options = []
    if params.get('logicalsize') is not None:
        options.append('--vdoLogicalSize=' + params['logicalsize'])
    if params.get('blockmapcachesize') is not None:
        options.append('--blockMapCacheSize=' + params['blockmapcachesize'])
    if params.get('readcache') == 'enabled':
        options.append('--readCache=enabled')
    if params.get('readcachesize') is not None:
        options.append('--readCacheSize=' + params['readcachesize'])
    if params.get('slabsize') is not None:
        options.append('--vdoSlabSize=' + params['slabsize'])
    if params.get('emulate512'):
        options.append('--emulate512=enabled')
    if params.get('indexmem') is not None:
        options.append('--indexMem=' + params['indexmem'])
    if params.get('indexmode') == 'sparse':
        options.append('--sparseIndex=enabled')
    if params.get('force'):
        options.append('--force')
    if params.get('ackthreads') is not None:
        options.append('--vdoAckThreads=' + params['ackthreads'])
    if params.get('biothreads') is not None:
        options.append('--vdoBioThreads=' + params['biothreads'])
    if params.get('cputhreads') is not None:
        options.append('--vdoCpuThreads=' + params['cputhreads'])
    if params.get('logicalthreads') is not None:
        options.append('--vdoLogicalThreads=' + params['logicalthreads'])
    if params.get('physicalthreads') is not None:
        options.append('--vdoPhysicalThreads=' + params['physicalthreads'])
    return options