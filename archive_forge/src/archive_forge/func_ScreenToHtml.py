from warnings import warn
import os
import pickle
import sys
import numpy
from rdkit import DataStructs
from rdkit.Dbase import DbModule
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import CompositeRun
from rdkit.ML.Data import DataUtils, SplitData
def ScreenToHtml(nGood, nBad, nRej, avgGood, avgBad, avgSkip, voteTable, imgDir='.', fullPage=1, skipImg=0, includeDefs=1):
    """ returns the text of a web page showing the screening details
#DOC
    **Arguments**

     - nGood: number of correct predictions

     - nBad:  number of incorrect predictions

     - nRej:  number of rejected predictions

     - avgGood: average correct confidence

     - avgBad:  average incorrect confidence

     - avgSkip: average rejected confidence

     - voteTable: vote table

     - imgDir: (optional) the directory to be used to hold the vote
       image (if constructed)

   **Returns**

     a string containing HTML

  """
    if type(nGood) == tuple:
        multModels = 1
    else:
        multModels = 0
    if fullPage:
        outTxt = ['<html><body>']
        outTxt.append('<center><h2>VOTE DETAILS</h2></center>')
    else:
        outTxt = []
    outTxt.append('<font>')
    if not skipImg:
        img = GetScreenImage(nGood, nBad, nRej)
        if img:
            if imgDir:
                imgFileName = '/'.join((imgDir, 'votes.png'))
            else:
                imgFileName = 'votes.png'
            img.save(imgFileName)
            outTxt.append('<center><img src="%s"></center>' % imgFileName)
    nPoss = len(voteTable)
    pureCounts = numpy.sum(voteTable, 1)
    accCounts = numpy.sum(voteTable, 0)
    pureVect = numpy.zeros(nPoss, float)
    accVect = numpy.zeros(nPoss, float)
    for i in range(nPoss):
        if pureCounts[i]:
            pureVect[i] = float(voteTable[i, i]) / pureCounts[i]
        if accCounts[i]:
            accVect[i] = float(voteTable[i, i]) / accCounts[i]
    outTxt.append('<center><table border=1>')
    outTxt.append('<tr><td></td>')
    for i in range(nPoss):
        outTxt.append('<th>%d</th>' % i)
    outTxt.append('<th>% Accurate</th>')
    outTxt.append('</tr>')
    for i in range(nPoss):
        outTxt.append('<tr><th>%d</th>' % i)
        for j in range(nPoss):
            if i == j:
                if not multModels:
                    outTxt.append('<td bgcolor="#A0A0FF">%d</td>' % voteTable[j, i])
                else:
                    outTxt.append('<td bgcolor="#A0A0FF">%.2f</td>' % voteTable[j, i])
            elif not multModels:
                outTxt.append('<td>%d</td>' % voteTable[j, i])
            else:
                outTxt.append('<td>%.2f</td>' % voteTable[j, i])
        outTxt.append('<td>%4.2f</td</tr>' % (100.0 * accVect[i]))
        if i == 0:
            outTxt.append('<th rowspan=%d>Predicted</th></tr>' % nPoss)
        else:
            outTxt.append('</tr>')
    outTxt.append('<tr><th>% Pure</th>')
    for i in range(nPoss):
        outTxt.append('<td>%4.2f</td>' % (100.0 * pureVect[i]))
    outTxt.append('</tr>')
    outTxt.append('<tr><td></td><th colspan=%d>Original</th>' % nPoss)
    outTxt.append('</table></center>')
    if not multModels:
        nTotal = nBad + nGood + nRej
        nClass = nBad + nGood
        if nClass:
            pctErr = 100.0 * float(nBad) / nClass
        else:
            pctErr = 0.0
        outTxt.append('<p>%d of %d examples were misclassified (%%%4.2f)' % (nBad, nGood + nBad, pctErr))
        if nRej > 0:
            pctErr = 100.0 * float(nBad) / (nGood + nBad + nRej)
            outTxt.append('<p>                %d of %d overall: (%%%4.2f)' % (nBad, nTotal, pctErr))
            pctRej = 100.0 * float(nRej) / nTotal
            outTxt.append('<p>%d of %d examples were rejected (%%%4.2f)' % (nRej, nTotal, pctRej))
        if nGood != 0:
            outTxt.append('<p>The correctly classified examples had an average confidence of %6.4f' % avgGood)
        if nBad != 0:
            outTxt.append('<p>The incorrectly classified examples had an average confidence of %6.4f' % avgBad)
        if nRej != 0:
            outTxt.append('<p>The rejected examples had an average confidence of %6.4f' % avgSkip)
    else:
        nTotal = nBad[0] + nGood[0] + nRej[0]
        nClass = nBad[0] + nGood[0]
        devClass = nBad[1] + nGood[1]
        if nClass:
            pctErr = 100.0 * float(nBad[0]) / nClass
            devPctErr = 100.0 * float(nBad[1]) / nClass
        else:
            pctErr = 0.0
            devPctErr = 0.0
        outTxt.append('<p>%.2f(%.2f) of %.2f(%.2f) examples were misclassified (%%%4.2f(%4.2f))' % (nBad[0], nBad[1], nClass, devClass, pctErr, devPctErr))
        if nRej > 0:
            pctErr = 100.0 * float(nBad[0]) / nTotal
            devPctErr = 100.0 * float(nBad[1]) / nTotal
            outTxt.append('<p>                %.2f(%.2f) of %d overall: (%%%4.2f(%4.2f))' % (nBad[0], nBad[1], nTotal, pctErr, devPctErr))
            pctRej = 100.0 * float(nRej[0]) / nTotal
            devPctRej = 100.0 * float(nRej[1]) / nTotal
            outTxt.append('<p>%.2f(%.2f) of %d examples were rejected (%%%4.2f(%4.2f))' % (nRej[0], nRej[1], nTotal, pctRej, devPctRej))
        if nGood != 0:
            outTxt.append('<p>The correctly classified examples had an average confidence of %6.4f(%.4f)' % avgGood)
        if nBad != 0:
            outTxt.append('<p>The incorrectly classified examples had an average confidence of %6.4f(%.4f)' % avgBad)
        if nRej != 0:
            outTxt.append('<p>The rejected examples had an average confidence of %6.4f(%.4f)' % avgSkip)
    outTxt.append('</font>')
    if includeDefs:
        txt = '\n    <p><b>Definitions:</b>\n    <ul>\n    <li> <i>% Pure:</i>  The percentage of, for example, known positives predicted to be positive.\n    <li> <i>% Accurate:</i>  The percentage of, for example, predicted positives that actually\n      are positive.\n    </ul>\n    '
        outTxt.append(txt)
    if fullPage:
        outTxt.append('</body></html>')
    return '\n'.join(outTxt)