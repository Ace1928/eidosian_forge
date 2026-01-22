import pickle
import re
from debian.deprecation import function_deprecated_by
class DB:
    """
    In-memory database mapping packages to tags and tags to packages.
    """

    def __init__(self):
        self.db = {}
        self.rdb = {}

    def read(self, input_data, tag_filter=None):
        """
        Read the database from a file.

        Example::
            # Read the system Debtags database
            db.read(open("/var/lib/debtags/package-tags", "r"))
        """
        self.db, self.rdb = read_tag_database_both_ways(input_data, tag_filter)

    def qwrite(self, file):
        """Quickly write the data to a pickled file"""
        pickle.dump(self.db, file)
        pickle.dump(self.rdb, file)

    def qread(self, file):
        """Quickly read the data from a pickled file"""
        self.db = pickle.load(file)
        self.rdb = pickle.load(file)

    def insert(self, pkg, tags):
        self.db[pkg] = tags.copy()
        for tag in tags:
            if tag in self.rdb:
                self.rdb[tag].add(pkg)
            else:
                self.rdb[tag] = set(pkg)

    def dump(self):
        output(self.db)

    def dump_reverse(self):
        output(self.rdb)
    dumpReverse = function_deprecated_by(dump_reverse)

    def reverse(self):
        """Return the reverse collection, sharing tagsets with this one"""
        res = DB()
        res.db = self.rdb
        res.rdb = self.db
        return res

    def facet_collection(self):
        """
        Return a copy of this collection, but replaces the tag names
        with only their facets.
        """
        fcoll = DB()
        tofacet = re.compile('^([^:]+).+')
        for pkg, tags in self.iter_packages_tags():
            ftags = {tofacet.sub('\\1', t) for t in tags}
            fcoll.insert(pkg, ftags)
        return fcoll
    facetCollection = function_deprecated_by(facet_collection)

    def copy(self):
        """
        Return a copy of this collection, with the tagsets copied as
        well.
        """
        res = DB()
        res.db = self.db.copy()
        res.rdb = self.rdb.copy()
        return res

    def reverse_copy(self):
        """
        Return the reverse collection, with a copy of the tagsets of
        this one.
        """
        res = DB()
        res.db = self.rdb.copy()
        res.rdb = self.db.copy()
        return res
    reverseCopy = function_deprecated_by(reverse_copy)

    def choose_packages(self, package_iter):
        """
        Return a collection with only the packages in package_iter,
        sharing tagsets with this one
        """
        res = DB()
        db = {}
        for pkg in package_iter:
            if pkg in self.db:
                db[pkg] = self.db[pkg]
        res.db = db
        res.rdb = reverse(db)
        return res
    choosePackages = function_deprecated_by(choose_packages)

    def choose_packages_copy(self, package_iter):
        """
        Return a collection with only the packages in package_iter,
        with a copy of the tagsets of this one
        """
        res = DB()
        db = {}
        for pkg in package_iter:
            db[pkg] = self.db[pkg]
        res.db = db
        res.rdb = reverse(db)
        return res
    choosePackagesCopy = function_deprecated_by(choose_packages_copy)

    def filter_packages(self, package_filter):
        """
        Return a collection with only those packages that match a
        filter, sharing tagsets with this one.  The filter will match
        on the package.
        """
        res = DB()
        db = {}
        for pkg in filter(package_filter, self.db.keys()):
            db[pkg] = self.db[pkg]
        res.db = db
        res.rdb = reverse(db)
        return res
    filterPackages = function_deprecated_by(filter_packages)

    def filter_packages_copy(self, filter_data):
        """
        Return a collection with only those packages that match a
        filter, with a copy of the tagsets of this one.  The filter
        will match on the package.
        """
        res = DB()
        db = {}
        for pkg in filter(filter_data, self.db.keys()):
            db[pkg] = self.db[pkg].copy()
        res.db = db
        res.rdb = reverse(db)
        return res
    filterPackagesCopy = function_deprecated_by(filter_packages_copy)

    def filter_packages_tags(self, package_tag_filter):
        """
        Return a collection with only those packages that match a
        filter, sharing tagsets with this one.  The filter will match
        on (package, tags).
        """
        res = DB()
        db = {}
        for pkg, _ in filter(package_tag_filter, self.db.items()):
            db[pkg] = self.db[pkg]
        res.db = db
        res.rdb = reverse(db)
        return res
    filterPackagesTags = function_deprecated_by(filter_packages_tags)

    def filter_packages_tags_copy(self, package_tag_filter):
        """
        Return a collection with only those packages that match a
        filter, with a copy of the tagsets of this one.  The filter
        will match on (package, tags).
        """
        res = DB()
        db = {}
        for pkg, _ in filter(package_tag_filter, self.db.items()):
            db[pkg] = self.db[pkg].copy()
        res.db = db
        res.rdb = reverse(db)
        return res
    filterPackagesTagsCopy = function_deprecated_by(filter_packages_tags_copy)

    def filter_tags(self, tag_filter):
        """
        Return a collection with only those tags that match a
        filter, sharing package sets with this one.  The filter will match
        on the tag.
        """
        res = DB()
        rdb = {}
        for tag in filter(tag_filter, self.rdb.keys()):
            rdb[tag] = self.rdb[tag]
        res.rdb = rdb
        res.db = reverse(rdb)
        return res
    filterTags = function_deprecated_by(filter_tags)

    def filter_tags_copy(self, tag_filter):
        """
        Return a collection with only those tags that match a
        filter, with a copy of the package sets of this one.  The
        filter will match on the tag.
        """
        res = DB()
        rdb = {}
        for tag in filter(tag_filter, self.rdb.keys()):
            rdb[tag] = self.rdb[tag].copy()
        res.rdb = rdb
        res.db = reverse(rdb)
        return res
    filterTagsCopy = function_deprecated_by(filter_tags_copy)

    def has_package(self, pkg):
        """Check if the collection contains the given package"""
        return pkg in self.db
    hasPackage = function_deprecated_by(has_package)

    def has_tag(self, tag):
        """Check if the collection contains packages tagged with tag"""
        return tag in self.rdb
    hasTag = function_deprecated_by(has_tag)

    def tags_of_package(self, pkg):
        """Return the tag set of a package"""
        return self.db[pkg] if pkg in self.db else set()
    tagsOfPackage = function_deprecated_by(tags_of_package)

    def packages_of_tag(self, tag):
        """Return the package set of a tag"""
        return self.rdb[tag] if tag in self.rdb else set()
    packagesOfTag = function_deprecated_by(packages_of_tag)

    def tags_of_packages(self, pkgs):
        """Return the set of tags that have all the packages in ``pkgs``"""
        return set.union(*(self.tags_of_package(p) for p in pkgs))
    tagsOfPackages = function_deprecated_by(tags_of_packages)

    def packages_of_tags(self, tags):
        """Return the set of packages that have all the tags in ``tags``"""
        return set.union(*(self.packages_of_tag(t) for t in tags))
    packagesOfTags = function_deprecated_by(packages_of_tags)

    def card(self, tag):
        """
        Return the cardinality of a tag
        """
        return len(self.rdb[tag]) if tag in self.rdb else 0

    def discriminance(self, tag):
        """
        Return the discriminance index if the tag.

        Th discriminance index of the tag is defined as the minimum
        number of packages that would be eliminated by selecting only
        those tagged with this tag or only those not tagged with this
        tag.
        """
        n = self.card(tag)
        tot = self.package_count()
        return min(n, tot - n)

    def iter_packages(self):
        """Iterate over the packages"""
        return self.db.keys()
    iterPackages = function_deprecated_by(iter_packages)

    def iter_tags(self):
        """Iterate over the tags"""
        return self.rdb.keys()
    iterTags = function_deprecated_by(iter_tags)

    def iter_packages_tags(self):
        """Iterate over 2-tuples of (pkg, tags)"""
        return self.db.items()
    iterPackagesTags = function_deprecated_by(iter_packages_tags)

    def iter_tags_packages(self):
        """Iterate over 2-tuples of (tag, pkgs)"""
        return self.rdb.items()
    iterTagsPackages = function_deprecated_by(iter_tags_packages)

    def package_count(self):
        """Return the number of packages"""
        return len(self.db)
    packageCount = function_deprecated_by(package_count)

    def tag_count(self):
        """Return the number of tags"""
        return len(self.rdb)
    tagCount = function_deprecated_by(tag_count)

    def ideal_tagset(self, tags):
        """
        Return an ideal selection of the top tags in a list of tags.

        Return the tagset made of the highest number of tags taken in
        consecutive sequence from the beginning of the given vector,
        that would intersect with the tagset of a comfortable amount
        of packages.

        Comfortable is defined in terms of how far it is from 7.
        """

        def score_fun(x):
            return float((x - 15) * (x - 15)) / x
        tagset = set()
        min_score = 3.0
        for i in range(len(tags)):
            pkgs = self.packages_of_tags(tags[:i + 1])
            card = len(pkgs)
            if card == 0:
                break
            score = score_fun(card)
            if score < min_score:
                min_score = score
                tagset = set(tags[:i + 1])
        if not tagset:
            return set(tags[:1])
        return tagset
    idealTagset = function_deprecated_by(ideal_tagset)

    def correlations(self):
        """
        Generate the list of correlation as a tuple (hastag, hasalsotag, score).

        Every tuple will indicate that the tag 'hastag' tends to also
        have 'hasalsotag' with a score of 'score'.
        """
        for pivot in self.iter_tags():
            with_ = self.filter_packages_tags(lambda pt: pivot in pt[1])
            without = self.filter_packages_tags(lambda pt: pivot not in pt[1])
            for tag in with_.iter_tags():
                if tag == pivot:
                    continue
                has = float(with_.card(tag)) / float(with_.package_count())
                hasnt = float(without.card(tag)) / float(without.package_count())
                yield (pivot, tag, has - hasnt)