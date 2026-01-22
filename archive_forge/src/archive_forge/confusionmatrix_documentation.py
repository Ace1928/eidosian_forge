from nltk.probability import FreqDist

        Tabulate the **recall**, **precision** and **f-measure**
        for each value in this confusion matrix.

        >>> reference = "DET NN VB DET JJ NN NN IN DET NN".split()
        >>> test = "DET VB VB DET NN NN NN IN DET NN".split()
        >>> cm = ConfusionMatrix(reference, test)
        >>> print(cm.evaluate())
        Tag | Prec.  | Recall | F-measure
        ----+--------+--------+-----------
        DET | 1.0000 | 1.0000 | 1.0000
         IN | 1.0000 | 1.0000 | 1.0000
         JJ | 0.0000 | 0.0000 | 0.0000
         NN | 0.7500 | 0.7500 | 0.7500
         VB | 0.5000 | 1.0000 | 0.6667
        <BLANKLINE>

        :param alpha: Ratio of the cost of false negative compared to false
            positives, as used in the f-measure computation. Defaults to 0.5,
            where the costs are equal.
        :type alpha: float
        :param truncate: If specified, then only show the specified
            number of values. Any sorting (e.g., sort_by_count)
            will be performed before truncation. Defaults to None
        :type truncate: int, optional
        :param sort_by_count: Whether to sort the outputs on frequency
            in the reference label. Defaults to False.
        :type sort_by_count: bool, optional
        :return: A tabulated recall, precision and f-measure string
        :rtype: str
        