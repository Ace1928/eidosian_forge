from warnings import warn
import pickle
import sys
import numpy
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import ScreenComposite
from rdkit.ML.Data import Stats
from rdkit.ML.DecTree import Tree, TreeUtils
def ErrorStats(conn, where, enrich=1):
    fields = 'overall_error,holdout_error,overall_result_matrix,' + 'holdout_result_matrix,overall_correct_conf,overall_incorrect_conf,' + 'holdout_correct_conf,holdout_incorrect_conf'
    try:
        data = conn.GetData(fields=fields, where=where)
    except Exception:
        import traceback
        traceback.print_exc()
        return None
    nPts = len(data)
    if not nPts:
        sys.stderr.write('no runs found\n')
        return None
    overall = numpy.zeros(nPts, float)
    overallEnrich = numpy.zeros(nPts, float)
    oCorConf = 0.0
    oInCorConf = 0.0
    holdout = numpy.zeros(nPts, float)
    holdoutEnrich = numpy.zeros(nPts, float)
    hCorConf = 0.0
    hInCorConf = 0.0
    overallMatrix = None
    holdoutMatrix = None
    for i in range(nPts):
        if data[i][0] is not None:
            overall[i] = data[i][0]
            oCorConf += data[i][4]
            oInCorConf += data[i][5]
        if data[i][1] is not None:
            holdout[i] = data[i][1]
            haveHoldout = 1
        else:
            haveHoldout = 0
        tmpOverall = 1.0 * eval(data[i][2])
        if enrich >= 0:
            overallEnrich[i] = ScreenComposite.CalcEnrichment(tmpOverall, tgt=enrich)
        if haveHoldout:
            tmpHoldout = 1.0 * eval(data[i][3])
            if enrich >= 0:
                holdoutEnrich[i] = ScreenComposite.CalcEnrichment(tmpHoldout, tgt=enrich)
        if overallMatrix is None:
            if data[i][2] is not None:
                overallMatrix = tmpOverall
            if haveHoldout and data[i][3] is not None:
                holdoutMatrix = tmpHoldout
        else:
            overallMatrix += tmpOverall
            if haveHoldout:
                holdoutMatrix += tmpHoldout
        if haveHoldout:
            hCorConf += data[i][6]
            hInCorConf += data[i][7]
    avgOverall = sum(overall) / nPts
    oCorConf /= nPts
    oInCorConf /= nPts
    overallMatrix /= nPts
    oSort = numpy.argsort(overall)
    oMin = overall[oSort[0]]
    overall -= avgOverall
    devOverall = numpy.sqrt(sum(overall ** 2) / (nPts - 1))
    res = {}
    res['oAvg'] = 100 * avgOverall
    res['oDev'] = 100 * devOverall
    res['oCorrectConf'] = 100 * oCorConf
    res['oIncorrectConf'] = 100 * oInCorConf
    res['oResultMat'] = overallMatrix
    res['oBestIdx'] = oSort[0]
    res['oBestErr'] = 100 * oMin
    if enrich >= 0:
        mean, dev = Stats.MeanAndDev(overallEnrich)
        res['oAvgEnrich'] = mean
        res['oDevEnrich'] = dev
    if haveHoldout:
        avgHoldout = sum(holdout) / nPts
        hCorConf /= nPts
        hInCorConf /= nPts
        holdoutMatrix /= nPts
        hSort = numpy.argsort(holdout)
        hMin = holdout[hSort[0]]
        holdout -= avgHoldout
        devHoldout = numpy.sqrt(sum(holdout ** 2) / (nPts - 1))
        res['hAvg'] = 100 * avgHoldout
        res['hDev'] = 100 * devHoldout
        res['hCorrectConf'] = 100 * hCorConf
        res['hIncorrectConf'] = 100 * hInCorConf
        res['hResultMat'] = holdoutMatrix
        res['hBestIdx'] = hSort[0]
        res['hBestErr'] = 100 * hMin
        if enrich >= 0:
            mean, dev = Stats.MeanAndDev(holdoutEnrich)
            res['hAvgEnrich'] = mean
            res['hDevEnrich'] = dev
    return res