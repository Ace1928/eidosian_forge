from nltk.classify import Senna
class SennaTagger(Senna):

    def __init__(self, path, encoding='utf-8'):
        super().__init__(path, ['pos'], encoding)

    def tag_sents(self, sentences):
        """
        Applies the tag method over a list of sentences. This method will return
        for each sentence a list of tuples of (word, tag).
        """
        tagged_sents = super().tag_sents(sentences)
        for i in range(len(tagged_sents)):
            for j in range(len(tagged_sents[i])):
                annotations = tagged_sents[i][j]
                tagged_sents[i][j] = (annotations['word'], annotations['pos'])
        return tagged_sents