import os
import sys
import time
import subprocess
from tempfile import NamedTemporaryFile
import low_index
from low_index import permutation_reps
from low_index import benchmark_util
def translate_to_gap(ex, output):
    output.write('info := "%s; index=%d";\n' % (ex['group'], ex['index']))
    all_relators = ex['short relators'] + ex['long relators']
    gap_relators = [benchmark_util.expand_relator(r) for r in all_relators]
    letters = 'abcdefghijklmnopqrstuvwxyz'
    generators = letters[:ex['rank']]
    output.write('F := FreeGroup(')
    output.write(', '.join(['"%s"' % g for g in generators]))
    output.write(');\n')
    for n, gen in enumerate(generators):
        output.write('%s := F.%d;\n' % (gen, n + 1))
    output.write('G := F / [\n')
    for relator in gap_relators:
        output.write('%s,\n' % relator)
    output.write('];\n')
    output.write('\nPrintFormatted("{}\\n", info);\nstart := NanosecondsSinceEpoch();\nans := Length(LowIndexSubgroupsFpGroup(G,%d));\nelapsed := Round(Float(NanosecondsSinceEpoch() - start) / 10000000.0)/100;\nPrintFormatted("{} subgroups\\n", ans);\nPrintFormatted("{} secs\\n", ViewString(elapsed));\n' % ex['index'])