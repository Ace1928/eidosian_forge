def update_myself():
    data_file = list(urlopen(DATATYPES_URL))
    datatypes = parse_datatypes(data_file)
    pseudos = parse_pseudos(data_file)
    keywords = parse_keywords(urlopen(KEYWORDS_URL))
    update_consts(__file__, 'DATATYPES', datatypes)
    update_consts(__file__, 'PSEUDO_TYPES', pseudos)
    update_consts(__file__, 'KEYWORDS', keywords)