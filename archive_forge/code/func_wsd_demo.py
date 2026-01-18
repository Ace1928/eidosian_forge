import math
import nltk.classify.util  # for accuracy & log_likelihood
from nltk.util import LazyMap
def wsd_demo(trainer, word, features, n=1000):
    import random
    from nltk.corpus import senseval
    print('Reading data...')
    global _inst_cache
    if word not in _inst_cache:
        _inst_cache[word] = [(i, i.senses[0]) for i in senseval.instances(word)]
    instances = _inst_cache[word][:]
    if n > len(instances):
        n = len(instances)
    senses = list({l for i, l in instances})
    print('  Senses: ' + ' '.join(senses))
    print('Splitting into test & train...')
    random.seed(123456)
    random.shuffle(instances)
    train = instances[:int(0.8 * n)]
    test = instances[int(0.8 * n):n]
    print('Training classifier...')
    classifier = trainer([(features(i), l) for i, l in train])
    print('Testing classifier...')
    acc = accuracy(classifier, [(features(i), l) for i, l in test])
    print('Accuracy: %6.4f' % acc)
    try:
        test_featuresets = [features(i) for i, n in test]
        pdists = classifier.prob_classify_many(test_featuresets)
        ll = [pdist.logprob(gold) for (name, gold), pdist in zip(test, pdists)]
        print('Avg. log likelihood: %6.4f' % (sum(ll) / len(test)))
    except NotImplementedError:
        pass
    return classifier