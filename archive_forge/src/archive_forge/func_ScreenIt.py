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
def ScreenIt(composite, indices, data, partialVote=0, voteTol=0.0, verbose=1, screenResults=None, goodVotes=None, badVotes=None, noVotes=None):
    """ screens a set of data using a composite model and prints out
             statistics about the screen.
#DOC
    The work of doing the screening and processing the results is
    handled by _DetailedScreen()_

  **Arguments**

    - composite:  the composite model to be used

    - data: the examples to be screened (a sequence of sequences)
       it's assumed that the last element in each example is its "value"

    - partialVote: (optional) toggles use of the threshold value in
      the screnning.

    - voteTol: (optional) the threshold to be used to decide whether or not a
      given prediction should be kept

    - verbose: (optional) sets degree of verbosity of the screening

    - screenResults: (optional) the results of screening the results
      (a sequence of 3-tuples in the format returned by
      _CollectResults()_).  If this is provided, the examples will not
      be screened again.

    - goodVotes,badVotes,noVotes: (optional)  if provided these should
      be lists (or anything supporting an _append()_ method) which
      will be used to pass the screening results back.


  **Returns**

    a 7-tuple:

      1) the number of good (correct) predictions

      2) the number of bad (incorrect) predictions

      3) the number of predictions skipped due to the _threshold_

      4) the average confidence in the good predictions

      5) the average confidence in the bad predictions

      6) the average confidence in the skipped predictions

      7) None

  """
    if goodVotes is None:
        goodVotes = []
    if badVotes is None:
        badVotes = []
    if noVotes is None:
        noVotes = []
    if not partialVote:
        voteTol = 0.0
    DetailedScreen(indices, data, composite, voteTol, screenResults=screenResults, goodVotes=goodVotes, badVotes=badVotes, noVotes=noVotes)
    nGood = len(goodVotes)
    goodAccum = 0.0
    for res, pred, conf, idx in goodVotes:
        goodAccum += conf
    misCount = len(badVotes)
    badAccum = 0.0
    for res, pred, conf, idx in badVotes:
        badAccum += conf
    nSkipped = len(noVotes)
    goodSkipped = 0
    badSkipped = 0
    skipAccum = 0.0
    for ans, pred, conf, idx in noVotes:
        skipAccum += conf
        if ans != pred:
            badSkipped += 1
        else:
            goodSkipped += 1
    nData = nGood + misCount + nSkipped
    if verbose:
        print('Total N Points:', nData)
    if partialVote:
        nCounted = nData - nSkipped
        if verbose:
            print('Misclassifications: %d (%%%4.2f)' % (misCount, 100.0 * float(misCount) / nCounted))
            print('N Skipped: %d (%%%4.2f)' % (nSkipped, 100.0 * float(nSkipped) / nData))
            print('\tGood Votes Skipped: %d (%%%4.2f)' % (goodSkipped, 100.0 * float(goodSkipped) / nSkipped))
            print('\tBad Votes Skipped: %d (%%%4.2f)' % (badSkipped, 100.0 * float(badSkipped) / nSkipped))
    elif verbose:
        print('Misclassifications: %d (%%%4.2f)' % (misCount, 100.0 * float(misCount) / nData))
        print('Average Correct Vote Confidence:   % 6.4f' % (goodAccum / (nData - misCount)))
        print('Average InCorrect Vote Confidence: % 6.4f' % (badAccum / misCount))
    avgGood = 0
    avgBad = 0
    avgSkip = 0
    if nGood:
        avgGood = goodAccum / nGood
    if misCount:
        avgBad = badAccum / misCount
    if nSkipped:
        avgSkip = skipAccum / nSkipped
    return (nGood, misCount, nSkipped, avgGood, avgBad, avgSkip, None)