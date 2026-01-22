from warnings import warn
import pickle
import sys
import numpy
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import ScreenComposite
from rdkit.ML.Data import Stats
from rdkit.ML.DecTree import Tree, TreeUtils
def ShowStats(statD, enrich=1):
    statD = statD.copy()
    statD['oBestIdx'] = statD['oBestIdx'] + 1
    txt = '\n# Error Statistics:\n\tOverall: %(oAvg)6.3f%% (%(oDev)6.3f)  %(oCorrectConf)4.1f/%(oIncorrectConf)4.1f\n\t\tBest: %(oBestIdx)d %(oBestErr)6.3f%%' % statD
    if 'hAvg' in statD:
        statD['hBestIdx'] = statD['hBestIdx'] + 1
        txt += '\n\tHoldout: %(hAvg)6.3f%% (%(hDev)6.3f)  %(hCorrectConf)4.1f/%(hIncorrectConf)4.1f\n\t\tBest: %(hBestIdx)d %(hBestErr)6.3f%%\n  ' % statD
    print(txt)
    print()
    print('# Results matrices:')
    print('\tOverall:')
    tmp = numpy.transpose(statD['oResultMat'])
    colCounts = sum(tmp)
    rowCounts = sum(tmp, 1)
    for i in range(len(tmp)):
        if rowCounts[i] == 0:
            rowCounts[i] = 1
        row = tmp[i]
        print('\t\t', end='')
        for j in range(len(row)):
            print('% 6.2f' % row[j], end='')
        print('\t| % 4.2f' % (100.0 * tmp[i, i] / rowCounts[i]))
    print('\t\t', end='')
    for i in range(len(tmp)):
        print('------', end='')
    print()
    print('\t\t', end='')
    for i in range(len(tmp)):
        if colCounts[i] == 0:
            colCounts[i] = 1
        print('% 6.2f' % (100.0 * tmp[i, i] / colCounts[i]), end='')
    print()
    if enrich > -1 and 'oAvgEnrich' in statD:
        print('\t\tEnrich(%d): %.3f (%.3f)' % (enrich, statD['oAvgEnrich'], statD['oDevEnrich']))
    if 'hResultMat' in statD:
        print('\tHoldout:')
        tmp = numpy.transpose(statD['hResultMat'])
        colCounts = sum(tmp)
        rowCounts = sum(tmp, 1)
        for i in range(len(tmp)):
            if rowCounts[i] == 0:
                rowCounts[i] = 1
            row = tmp[i]
            print('\t\t', end='')
            for j in range(len(row)):
                print('% 6.2f' % row[j], end='')
            print('\t| % 4.2f' % (100.0 * tmp[i, i] / rowCounts[i]))
        print('\t\t', end='')
        for i in range(len(tmp)):
            print('------', end='')
        print()
        print('\t\t', end='')
        for i in range(len(tmp)):
            if colCounts[i] == 0:
                colCounts[i] = 1
            print('% 6.2f' % (100.0 * tmp[i, i] / colCounts[i]), end='')
        print()
        if enrich > -1 and 'hAvgEnrich' in statD:
            print('\t\tEnrich(%d): %.3f (%.3f)' % (enrich, statD['hAvgEnrich'], statD['hDevEnrich']))
    return