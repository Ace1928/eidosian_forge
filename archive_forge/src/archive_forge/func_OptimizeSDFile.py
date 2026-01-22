import os
import tempfile
import time
import re
def OptimizeSDFile(inFileName, outFileName, problemFileName='problems.sdf', restartEvery=20):
    """  optimizes the structure of every molecule in the input SD file

    **Arguments**

      - inFileName: name of the input SD file

      - outFileName: name of the output SD file

      - problemFileName: (optional) name of the SD file used to store molecules which
        fail during the optimization process

      - restartEvery: (optional)  Chem3D will be shut down and restarted
        every _restartEvery_ molecules to try and keep core leaks under control

  """
    inFile = open(inFileName, 'r')
    outFile = open(outFileName, 'w+')
    problemFile = None
    props = {}
    lines = []
    nextLine = inFile.readline()
    skip = 0
    nDone = 0
    t1 = time.time()
    while nextLine != '':
        if nextLine.find('M  END') != -1:
            lines.append(nextLine)
            molBlock = ''.join(lines)
            try:
                newMolBlock = Add3DCoordsToMol(molBlock, 'chemical/mdl-molfile', props=props)
            except Exception:
                badBlock = molBlock
                skip = 1
                lines = []
            else:
                skip = 0
                lines = [newMolBlock]
        elif nextLine.find('$$$$') != -1:
            t2 = time.time()
            nDone += 1
            print('finished molecule %d in %f seconds' % (nDone, time.time() - t1))
            t1 = time.time()
            if nDone % restartEvery == 0:
                CloseChem3D()
                StartChem3D()
                outFile.close()
                outFile = open(outFileName, 'a')
            if not skip:
                for prop in props.keys():
                    lines.append('> <%s>\n%f\n\n' % (prop, props[prop]))
                lines.append(nextLine)
                outFile.write(''.join(lines))
                lines = []
            else:
                skip = 0
                lines.append(nextLine)
                if problemFile is None:
                    problemFile = open(problemFileName, 'w+')
                problemFile.write(badBlock)
                problemFile.write(''.join(lines))
                lines = []
        else:
            lines.append(nextLine)
        nextLine = inFile.readline()
    outFile.close()
    if problemFile is not None:
        problemFile.close()